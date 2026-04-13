"""Tests for config.AgentConfig."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from config import AgentConfig


class AgentConfigTests(unittest.TestCase):
    def test_from_args_copies_fields(self) -> None:
        ns = argparse.Namespace(
            model="m",
            host="http://h:1",
            workdir=".",
            ctx=16384,
        )
        cfg = AgentConfig.from_args(ns)
        self.assertEqual(cfg.model, "m")
        self.assertEqual(cfg.host, "http://h:1")
        self.assertEqual(cfg.ctx, 16384)

    def test_from_args_resolves_workdir(self) -> None:
        ns = argparse.Namespace(
            model="m", host="h", workdir=".", ctx=1024
        )
        cfg = AgentConfig.from_args(ns)
        self.assertTrue(cfg.workdir.is_absolute())
        self.assertEqual(cfg.workdir, Path(".").resolve())

    def test_frozen(self) -> None:
        cfg = AgentConfig(model="m", host="h", workdir=Path("."), ctx=1)
        with self.assertRaises(Exception):  # FrozenInstanceError
            cfg.model = "x"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
