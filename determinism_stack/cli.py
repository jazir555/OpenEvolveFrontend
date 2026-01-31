"""CLI for deterministic stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .llm import LLMConfig, build_llm
from .pipeline import DeterministicPipeline, DeterminismConfig
from .monitoring import cloud_consensus


def _load_schema(schema_path: Optional[str], schema_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if schema_path:
        return json.loads(Path(schema_path).read_text(encoding="utf-8"))
    if schema_json:
        return json.loads(schema_json)
    return None


def build_pipeline(args: argparse.Namespace) -> DeterministicPipeline:
    llm = None
    if args.provider and args.model:
        llm = build_llm(
            LLMConfig(
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                seed=args.seed,
                device=args.device,
                dtype=args.dtype,
            )
        )
    config = DeterminismConfig(
        detllm_backend=args.detllm_backend,
        detllm_model=args.detllm_model,
        detllm_mode=args.detllm_mode,
        lmql_model=args.lmql_model,
        filter_intensity=args.filter_intensity,
    )
    return DeterministicPipeline(llm=llm, config=config)


def cmd_generate(args: argparse.Namespace) -> None:
    pipeline = build_pipeline(args)
    schema = _load_schema(args.schema_path, args.schema_json)
    result = pipeline.generate_with_all_layers(args.prompt, schema=schema, constraints=args.constraints, context_document=args.document)
    print(json.dumps(result.__dict__, indent=2, default=str))


def cmd_check(args: argparse.Namespace) -> None:
    pipeline = build_pipeline(args)
    result = pipeline.reproducibility.check(
        prompt=args.prompt,
        llm=pipeline.llm,
        tier=args.tier,
        runs=args.runs,
        backend=args.detllm_backend,
        model=args.detllm_model,
    )
    print(json.dumps(result, indent=2))


def cmd_consensus(args: argparse.Namespace) -> None:
    pipeline = build_pipeline(args)
    result = cloud_consensus(args.prompt, runs=args.runs, threshold=args.threshold, llm=pipeline.llm)
    print(json.dumps(result, indent=2))


def cmd_decompose(args: argparse.Namespace) -> None:
    pipeline = build_pipeline(args)
    result = pipeline.decomposer.decompose(args.prompt)
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic LLM Stack CLI")
    parser.add_argument("--provider", help="LLM provider (openai, anthropic, google, hf)")
    parser.add_argument("--model", help="LLM model name or path")
    parser.add_argument("--api-key", help="API key for cloud providers")
    parser.add_argument("--base-url", help="Base URL for cloud providers")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--detllm-backend", default=None)
    parser.add_argument("--detllm-model", default=None)
    parser.add_argument("--detllm-mode", default="auto")
    parser.add_argument("--lmql-model", default=None)
    parser.add_argument("--filter-intensity", type=float, default=0.5)

    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate with deterministic stack")
    gen.add_argument("prompt")
    gen.add_argument("--schema-path")
    gen.add_argument("--schema-json")
    gen.add_argument("--constraints")
    gen.add_argument("--document")
    gen.set_defaults(func=cmd_generate)

    check = sub.add_parser("check", help="Run reproducibility check")
    check.add_argument("prompt")
    check.add_argument("--tier", type=int, default=2)
    check.add_argument("--runs", type=int, default=3)
    check.set_defaults(func=cmd_check)

    consensus = sub.add_parser("consensus", help="Run cloud consensus")
    consensus.add_argument("prompt")
    consensus.add_argument("--runs", type=int, default=5)
    consensus.add_argument("--threshold", type=float, default=0.6)
    consensus.set_defaults(func=cmd_consensus)

    decompose = sub.add_parser("decompose", help="Decompose prompt")
    decompose.add_argument("prompt")
    decompose.set_defaults(func=cmd_decompose)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
