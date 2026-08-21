"""
Offline tests for previously-stubbed configuration flags.

Covers:
  * Config round-trip (from_dict / to_dict) for the flags.
  * build_meta_prompt() transforming the base prompt.
  * Distributed evaluation running across a process pool.
  * Resource limits being forwarded to SecureCodeExecutor.
"""

import asyncio
import os

import pytest

from openevolve.config import Config, EvaluatorConfig
from openevolve.evaluator import Evaluator
from openevolve.prompt import build_meta_prompt


def test_flags_round_trip():
    cfg_dict = {
        "prompt": {"use_meta_prompting": True, "meta_prompt_weight": 0.25},
        "evaluator": {
            "memory_limit_mb": 256,
            "cpu_limit": 1.5,
            "secure_execution": True,
            "distributed": True,
        },
    }
    cfg = Config.from_dict(cfg_dict)
    assert cfg.prompt.use_meta_prompting is True
    assert cfg.prompt.meta_prompt_weight == 0.25
    assert cfg.evaluator.memory_limit_mb == 256
    assert cfg.evaluator.cpu_limit == 1.5
    assert cfg.evaluator.secure_execution is True
    assert cfg.evaluator.distributed is True

    out = cfg.to_dict()
    assert out["prompt"]["use_meta_prompting"] is True
    assert out["evaluator"]["memory_limit_mb"] == 256
    assert out["evaluator"]["distributed"] is True

    # Defaults keep everything OFF so existing behavior is unchanged.
    default = Config()
    assert default.prompt.use_meta_prompting is False
    assert default.evaluator.memory_limit_mb is None
    assert default.evaluator.cpu_limit is None
    assert default.evaluator.secure_execution is False
    assert default.evaluator.distributed is False


def test_build_meta_prompt_transforms():
    base = "Write a function to add two numbers."
    wrapped = build_meta_prompt(base, weight=0.3)
    assert base in wrapped
    assert "BEGIN TASK INSTRUCTIONS" in wrapped
    assert "0.30" in wrapped
    assert wrapped != base


def _make_eval_file():
    fd, path = __import__("tempfile").mkstemp(suffix=".py", prefix="openevolve_eval_")
    with os.fdopen(fd, "w") as f:
        f.write("def evaluate(program_path):\n    return {'score': 1.0}\n")
    return path


def test_distributed_evaluation_runs():
    eval_file = _make_eval_file()
    try:
        config = EvaluatorConfig(distributed=True, parallel_evaluations=1)
        evaluator = Evaluator(config, eval_file)
        programs = [
            ("def add(a,b):\n    return a+b\n", f"p{i}") for i in range(3)
        ]
        results = asyncio.run(evaluator.evaluate_multiple(programs))
        assert len(results) == 3
        for r in results:
            assert r["score"] == 1.0
    finally:
        os.unlink(eval_file)


def test_resource_limits_passed_to_executor():
    import openevolve.secure_executor as se

    eval_file = _make_eval_file()
    captured = {}
    orig_init = se.SecureCodeExecutor.__init__

    def fake_init(self, config=None):
        captured["config"] = config
        orig_init(self, config)

    async def fake_execute(self, source, stdin=None, env=None):
        return se.SecureExecResult(
            stdout='{"__result__": {"score": 1.0}}', returncode=0
        )

    orig_execute = se.SecureCodeExecutor.execute_code
    se.SecureCodeExecutor.__init__ = fake_init
    se.SecureCodeExecutor.execute_code = fake_execute
    try:
        config = EvaluatorConfig(
            secure_execution=True,
            memory_limit_mb=256,
            cpu_limit=1.5,
        )
        evaluator = Evaluator(config, eval_file)
        result = asyncio.run(evaluator.evaluate_program("x=1\n", "p1"))
        assert result["score"] == 1.0
        assert captured["config"] is not None
        assert captured["config"].memory_limit_mb == 256
        assert captured["config"].cpu_time_limit == 1.5
    finally:
        se.SecureCodeExecutor.__init__ = orig_init
        se.SecureCodeExecutor.execute_code = orig_execute
        os.unlink(eval_file)
