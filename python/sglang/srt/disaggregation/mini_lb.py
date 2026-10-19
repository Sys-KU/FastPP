"""
A minimal router for PD disaggregation, for experiments.

Every /generate request is sent to a prefill server and a decode server with the
same bootstrap room. The prefill server computes the prompt KV and the first
token and sends them to the decode server, which generates the rest; the router
returns the response of the decode server.

Usage:
    python -m sglang.srt.disaggregation.mini_lb \
        --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --port 8000
"""

import argparse
import asyncio
import itertools
import logging
import random
import urllib.parse
from typing import Dict, List, Tuple

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)


class MiniLoadBalancer:
    def __init__(self, prefill_urls: List[str], decode_urls: List[str]):
        self.prefill_urls = prefill_urls
        self.decode_urls = decode_urls
        self.prefill_cycle = itertools.cycle(prefill_urls)
        self.decode_cycle = itertools.cycle(decode_urls)
        # prefill url -> (bootstrap host, bootstrap port)
        self.bootstrap: Dict[str, Tuple[str, int]] = {}
        self.session: aiohttp.ClientSession = None
        self.background_tasks = set()

    async def start(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None),
            # Close idle connections before the servers do (uvicorn keeps them for
            # 5 s); otherwise a request can be sent on a connection being closed.
            connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=4),
        )
        for url in self.prefill_urls + self.decode_urls:
            async with self.session.get(f"{url}/get_server_info") as resp:
                resp.raise_for_status()
                info = await resp.json()
            expected = "prefill" if url in self.prefill_urls else "decode"
            if info.get("disaggregation_mode") != expected:
                raise RuntimeError(
                    f"{url} must be launched with --disaggregation-mode {expected}"
                )
            if expected == "prefill":
                host = urllib.parse.urlparse(url).hostname
                self.bootstrap[url] = (host, info["disaggregation_bootstrap_port"])
        logger.info(f"Prefill servers: {self.bootstrap}, decode servers: {self.decode_urls}")

    async def stop(self):
        await self.session.close()

    def select_pair(self) -> Tuple[str, str]:
        return next(self.prefill_cycle), next(self.decode_cycle)

    def with_bootstrap(self, request_data: dict, prefill_url: str) -> dict:
        host, port = self.bootstrap[prefill_url]
        text, input_ids = request_data.get("text"), request_data.get("input_ids")
        if isinstance(text, list):
            batch_size = len(text)
        elif input_ids and isinstance(input_ids[0], list):
            batch_size = len(input_ids)
        else:
            batch_size = None

        def new_room():
            return random.randint(0, 2**63 - 1)

        return {
            **request_data,
            "bootstrap_host": host,
            "bootstrap_port": port,
            "bootstrap_room": (
                new_room()
                if batch_size is None
                else [new_room() for _ in range(batch_size)]
            ),
        }

    async def send_prefill(self, prefill_url: str, request_data: dict):
        # The output of the prefill server is only the first token; drop it.
        try:
            async with self.session.post(
                f"{prefill_url}/generate", json={**request_data, "stream": False}
            ) as resp:
                body = await resp.read()
                if resp.status != 200:
                    logger.error(
                        f"Prefill server error {resp.status}: {body[:500]!r}"
                    )
        except Exception as e:
            logger.error(f"Prefill request failed: {e}")

    def start_prefill(self, prefill_url: str, request_data: dict):
        task = asyncio.create_task(self.send_prefill(prefill_url, request_data))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def fan_out(self, method: str, path: str):
        async def call(url):
            async with self.session.request(method, f"{url}{path}") as resp:
                return resp.status

        statuses = await asyncio.gather(
            *[call(url) for url in self.prefill_urls + self.decode_urls]
        )
        return all(status == 200 for status in statuses)


app = FastAPI()
lb: MiniLoadBalancer = None


@app.on_event("startup")
async def startup():
    await lb.start()


@app.on_event("shutdown")
async def shutdown():
    await lb.stop()


@app.get("/health")
async def health():
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate():
    ok = await lb.fan_out("GET", "/health_generate")
    return Response(status_code=200 if ok else 503)


@app.post("/flush_cache")
async def flush_cache():
    ok = await lb.fan_out("POST", "/flush_cache")
    return Response(status_code=200 if ok else 500)


@app.get("/get_server_info")
async def get_server_info():
    infos = {}
    for kind, urls in (("prefill", lb.prefill_urls), ("decode", lb.decode_urls)):
        infos[kind] = []
        for url in urls:
            async with lb.session.get(f"{url}/get_server_info") as resp:
                infos[kind].append(await resp.json())
    return infos


@app.get("/get_model_info")
async def get_model_info():
    async with lb.session.get(f"{lb.decode_urls[0]}/get_model_info") as resp:
        return ORJSONResponse(await resp.json(), status_code=resp.status)


@app.get("/v1/models")
async def get_models():
    async with lb.session.get(f"{lb.decode_urls[0]}/v1/models") as resp:
        return ORJSONResponse(await resp.json(), status_code=resp.status)


@app.post("/generate")
async def generate(request: Request):
    request_data = orjson.loads(await request.body())
    prefill_url, decode_url = lb.select_pair()
    request_data = lb.with_bootstrap(request_data, prefill_url)
    lb.start_prefill(prefill_url, request_data)

    if request_data.get("stream", False):
        # Open the decode stream before returning, so that an error status of
        # the decode server reaches the client.
        resp = await lb.session.post(f"{decode_url}/generate", json=request_data)
        if resp.status != 200:
            body = await resp.read()
            resp.release()
            return Response(content=body, status_code=resp.status)

        async def stream():
            try:
                async for chunk in resp.content.iter_any():
                    yield chunk
            finally:
                resp.release()

        return StreamingResponse(stream(), media_type="text/event-stream")

    async with lb.session.post(f"{decode_url}/generate", json=request_data) as resp:
        body = await resp.read()
        return Response(
            content=body, status_code=resp.status, media_type="application/json"
        )


def main():
    global lb
    parser = argparse.ArgumentParser(description="Minimal PD disaggregation router")
    parser.add_argument(
        "--prefill", required=True, help="Comma-separated URLs of prefill servers"
    )
    parser.add_argument(
        "--decode", required=True, help="Comma-separated URLs of decode servers"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    lb = MiniLoadBalancer(args.prefill.split(","), args.decode.split(","))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
