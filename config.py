"""Central configuration — single source of truth for all agent settings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for the coding agent."""

    model: str
    host: str
    workdir: Path
    ctx: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> AgentConfig:
        """Build config from parsed CLI arguments."""
        return cls(
            model=args.model,
            host=args.host,
            workdir=Path(args.workdir).resolve(),
            ctx=args.ctx,
        )
