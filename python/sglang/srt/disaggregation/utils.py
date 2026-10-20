"""Shared definitions for PD (prefill/decode) disaggregation."""

import enum
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from sglang.srt.managers.io_struct import TokenizedGenerateReqInput


class DisaggregationMode(str, enum.Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


def get_transfer_port(bootstrap_port: int, pp_rank: int, tp_rank: int, tp_size: int):
    """The port where prefill rank (pp_rank, tp_rank) accepts its decode peer."""
    return bootstrap_port + pp_rank * tp_size + tp_rank


def rebootstrap_endpoint(bootstrap_port: int) -> str:
    """The IPC path where rank 0 of the prefill server with this bootstrap port takes
    the requests that decode servers retracted."""
    return os.path.join(tempfile.gettempdir(), f"sglang_pd_rebootstrap_{bootstrap_port}")


def next_bootstrap_room(room: int) -> int:
    """The bootstrap room of a retracted request sent back to the prefill server.

    Every rank of the decode server derives the same room (a step of a 63-bit LCG),
    and it collides with another room as rarely as a random one."""
    return (room * 6364136223846793005 + 1442695040888963407) % (1 << 63)


@dataclass
class DisaggRebootstrapReq:
    """A request that a decode server retracted, sent back to the prefill server.

    The prefill server computes the KV of the prompt and of the outputs so far again
    and sends it with the next token, as for the prompt of a new request: the decode
    server never prefills. Nothing is streamed from the prefill server."""

    req: TokenizedGenerateReqInput


@dataclass
class DisaggControl:
    """Decisions of scheduler rank 0, broadcast with the requests of an iteration.

    Transfers progress asynchronously on every rank, but the scheduling state is
    replicated across ranks, so rank 0 decides when a request moves on and all
    ranks apply the decision at the same iteration.
    """

    # Prefill: every rank has sent the KV of these rooms.
    sent: List[int] = field(default_factory=list)
    # Decode: every rank has received the KV of these rooms (room, first token id).
    received: List[Tuple[int, int]] = field(default_factory=list)
    # Decode: (room, reason) of transfers to stop. The KV is freed only when every
    # rank has stopped writing it.
    cancel: List[Tuple[int, str]] = field(default_factory=list)
    # Decode: rooms that every rank has stopped receiving.
    released: List[int] = field(default_factory=list)
    # Prefill: (room, reason) of failed requests.
    failed: List[Tuple[int, str]] = field(default_factory=list)

    def __bool__(self):
        return bool(
            self.sent
            or self.received
            or self.cancel
            or self.released
            or self.failed
        )


class TransferConsensus:
    """Collects per-rank transfer completions on scheduler rank 0."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.done_ranks: Dict[int, Set[int]] = {}
        self.first_token: Dict[int, Optional[int]] = {}

    def add(self, room: int, rank: int, first_token: Optional[int]):
        self.done_ranks.setdefault(room, set()).add(rank)
        self.first_token.setdefault(room, first_token)

    def discard(self, room: int):
        self.done_ranks.pop(room, None)
        self.first_token.pop(room, None)

    def pop_complete(self) -> List[Tuple[int, Optional[int]]]:
        rooms = [
            room
            for room, ranks in self.done_ranks.items()
            if len(ranks) == self.world_size
        ]
        complete = [(room, self.first_token[room]) for room in rooms]
        for room in rooms:
            self.discard(room)
        return complete
