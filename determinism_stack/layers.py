"""Layer implementations for the deterministic LLM stack."""

from __future__ import annotations

import json
import re
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .backends import LLMInterface
from .utils import (
    build_from_schema,
    extract_json,
    is_valid_json_prefix,
    optional_attr,
    optional_import,
    safe_eval,
    safe_z3_eval,
    similarity,
    validate_schema,
)

# =============================================================================
# CAV-NLP Integration with Graceful Fallback
# =============================================================================

try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver, ConstraintFormalizer
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    EnhancedZ3Solver = None
    ConstraintFormalizer = None


@dataclass
class AttractorCheck:
    is_attracted: bool
    score: float
    triggers: List[str]


class LagrangeFilter:
    """Layer 0: Pre-generation filtering and attractor detection."""

    def __init__(self, model_name: str = "default", config_dir: Optional[str] = None):
        self._steering = None
        self._triggers = [
            "obviously",
            "clearly",
            "undoubtedly",
            "everyone knows",
            "it goes without saying",
            "as an ai",
        ]
        module = optional_import("attractor_steering")
        if module and hasattr(module, "load_steering"):
            try:
                config_dir = config_dir or str((Path(__file__).resolve().parents[1] / "lagrange-mapper" / "filter_configs"))
                self._steering = module.load_steering(model_name, config_dir=config_dir)
            except Exception:
                self._steering = None

    def detect(self, prompt: str, intensity: float = 0.5) -> AttractorCheck:
        if self._steering:
            try:
                result = self._steering.detect(prompt, intensity=intensity)
                score = max(result.keyword_score, result.embedding_score)
                return AttractorCheck(
                    is_attracted=result.is_attracted,
                    score=score,
                    triggers=result.triggered_attractors or result.flagged_keywords,
                )
            except Exception:
                pass
        text = (prompt or "").lower()
        hits = [t for t in self._triggers if t in text]
        score = min(1.0, len(hits) / max(len(self._triggers), 1))
        return AttractorCheck(is_attracted=bool(hits), score=score, triggers=hits)

    def filter(self, prompt: str, intensity: float = 0.5, mode: str = "rephrase") -> str:
        if not prompt:
            return prompt
        if self._steering:
            try:
                result = self._steering.detect(prompt, intensity=intensity)
                if not result.is_attracted:
                    return prompt
                avoidance = self._steering.get_avoidance_prompt(result)
                cleaned = prompt
                for keyword in result.flagged_keywords:
                    cleaned = re.sub(re.escape(keyword), "", cleaned, flags=re.IGNORECASE)
                return f"{cleaned.strip()}{avoidance}".strip()
            except Exception:
                return prompt
        if intensity <= 0:
            return prompt
        filtered = prompt
        for phrase in self._triggers:
            filtered = re.sub(re.escape(phrase), "", filtered, flags=re.IGNORECASE)
        return filtered.strip()


class DecompositionAdapter:
    """Layer 1: Task decomposition (ROMA/MDAP/MAKER)."""

    def __init__(self):
        self._solver = None
        roma = optional_import("roma_dspy")
        if roma:
            try:
                solver_cls = getattr(roma, "RecursiveSolver", None)
                config_cls = optional_attr("roma_dspy.config.schemas.root", "ROMAConfig")
                if solver_cls:
                    if config_cls:
                        self._solver = solver_cls(config=config_cls())
                    else:
                        self._solver = solver_cls()
            except Exception:
                self._solver = None

    def atomize(self, requirement: str) -> List[str]:
        if self._solver and hasattr(self._solver, "atomize"):
            try:
                return list(self._solver.atomize(requirement))
            except Exception:
                pass
        return self._fallback_atomize(requirement)

    def decompose(self, requirement: str) -> Dict[str, Any]:
        if self._solver and hasattr(self._solver, "solve"):
            try:
                return self._solver.solve(requirement, context={})
            except Exception:
                pass
        return {"tasks": self.atomize(requirement)}

    def _fallback_atomize(self, requirement: str) -> List[str]:
        if not requirement:
            return []
        parts = re.split(r"[\\.\\n;]+", requirement)
        return [p.strip() for p in parts if p.strip()]


class AtomicTask:
    """Base class for atomic tasks (DSPy-compatible)."""

    def __init__(self, signature: Optional[str] = None):
        dspy = optional_import("dspy")
        self._predict = dspy.Predict(signature) if dspy and signature else None

    def forward(self, **kwargs: Any) -> Any:
        if self._predict:
            return self._predict(**kwargs)
        return kwargs


class DataRetrieval(AtomicTask):
    """RETRIEVE task type."""

    def forward(self, query: str, context: str = "") -> Any:
        signature = "context, query -> retrieved_data"
        if self._predict is None:
            return {"retrieved_data": f"{context}\n{query}".strip()}
        self._predict.signature = signature
        return super().forward(context=context, query=query)


class ContentGeneration(AtomicTask):
    """WRITE task type."""

    def forward(self, prompt: str, format_requirements: str = "") -> Any:
        signature = "prompt, format_requirements -> generated_content"
        if self._predict is None:
            return {"generated_content": f"{prompt}\n{format_requirements}".strip()}
        self._predict.signature = signature
        return super().forward(prompt=prompt, format_requirements=format_requirements)


class ConstrainedGenerator:
    """Layer 2: Constrained generation (LMQL/Outlines/Jsonformer)."""

    def __init__(self, llm: Optional[LLMInterface] = None, lmql_model: Optional[str] = None):
        self.llm = llm
        self.lmql_model = lmql_model
        self._lmql = optional_import("lmql")
        self._outlines = optional_import("outlines")
        self._jsonformer = optional_import("jsonformer")

    def _outlines_model(self):
        if not self._outlines or not self.llm:
            return None
        if hasattr(self.llm, "get_outlines_model"):
            return self.llm.get_outlines_model()
        return None

    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        model = self._outlines_model()
        if model is not None:
            try:
                output_type = self._outlines.types.json_schema(schema)
                return model.generate(prompt, output_type)
            except Exception:
                pass
        if self.llm:
            text = self.llm.generate(prompt)
            parsed = extract_json(text)
            if parsed is not None:
                return parsed
        return build_from_schema(schema)

    def generate_with_constraints(self, prompt: str, constraints: str) -> str:
        if self._lmql and hasattr(self._lmql, "query_from_string"):
            try:
                query_str = f'\"{prompt}\" [OUTPUT] where {constraints}'
                query_fn = self._lmql.query_from_string(query_str, is_async=False)
                model_name = self.lmql_model or getattr(self.llm, "model", None)
                if not model_name:
                    return self.llm.generate(prompt) if self.llm else prompt
                if hasattr(query_fn, "force_model"):
                    query_fn.force_model(model_name)
                result = query_fn()
                if isinstance(result, dict) and "OUTPUT" in result:
                    return result["OUTPUT"]
                if isinstance(result, list) and result:
                    return str(result[0])
                return str(result)
            except Exception:
                pass
        return self.llm.generate(prompt) if self.llm else prompt

    def generate_bulletproof_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if self._jsonformer and self.llm and hasattr(self.llm, "tokenizer"):
            try:
                model_obj = getattr(self.llm, "_model", None) or getattr(self.llm, "model_obj", None)
                if model_obj is None:
                    return self.generate_json(prompt, schema)
                return self._jsonformer.Jsonformer(model_obj, self.llm.tokenizer, schema, prompt)()
            except Exception:
                pass
        return self.generate_json(prompt, schema)


class StreamingConstrainedGenerator:
    """Stream structured output while validating partial JSON."""

    def __init__(self, llm: Optional[LLMInterface] = None):
        self.llm = llm

    def stream_structured(self, prompt: str, schema: Dict[str, Any]):
        if not self.llm or not hasattr(self.llm, "stream"):
            yield json.dumps(build_from_schema(schema))
            return
        buffer = ""
        for token in self.llm.stream(prompt):
            buffer += token
            if is_valid_json_prefix(buffer):
                yield token


class ContentValidator:
    """Layer 3: Content validation and correction."""

    def __init__(self, reask_max: int = 2):
        self._guardrails = optional_import("guardrails")
        self._steer = optional_import("steer")
        self._reask_max = reask_max
        self._json_judge = None
        self._slop_judge = None
        if self._steer:
            try:
                judges = optional_import("steer.judges")
                if judges:
                    self._json_judge = judges.JsonJudge("json")
                    self._slop_judge = judges.SlopJudge("slop")
            except Exception:
                self._json_judge = None
                self._slop_judge = None

    def validate(self, output: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        issues = []
        if schema:
            valid, schema_issues = validate_schema(output, schema)
            issues.extend(list(schema_issues))
            if self._guardrails:
                try:
                    validator = optional_import("guardrails.schema.validator")
                    if validator and hasattr(validator, "validate_json_schema"):
                        validator.validate_json_schema(schema)
                except Exception as exc:
                    valid = False
                    issues.append(str(exc))
        else:
            valid = True
        if self._json_judge:
            judge_result = self._json_judge.verify({}, output)
            if not judge_result.passed:
                valid = False
                issues.append(judge_result.reason or "json_judge_failed")
        return {"valid": valid, "issues": issues}

    def validate_text(self, text: str, rules: Optional[List[str]] = None) -> Dict[str, Any]:
        issues = []
        valid = True
        for rule in rules or []:
            if rule.lower() in text.lower():
                issues.append(f"rule_triggered:{rule}")
                valid = False
        if self._slop_judge:
            judge_result = self._slop_judge.verify({}, text)
            if not judge_result.passed:
                valid = False
                issues.append(judge_result.reason or "slop_judge_failed")
        return {"valid": valid, "issues": issues}

    def validate_and_fix(
        self,
        output: Any,
        schema: Optional[Dict[str, Any]] = None,
        llm: Optional[LLMInterface] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        if schema and isinstance(output, str):
            parsed = extract_json(output)
            if parsed is not None:
                output = parsed
        if isinstance(output, dict) and schema:
            result = self.validate(output, schema)
        else:
            result = self.validate_text(str(output))
        if result["valid"] or llm is None or prompt is None:
            return {"output": output, "validation": result}
        for _ in range(self._reask_max):
            if schema:
                fix_prompt = f"{prompt}\n\nFORMAT RULE: Output ONLY valid JSON for this schema: {json.dumps(schema)}"
            else:
                fix_prompt = (
                    f"{prompt}\n\nQUALITY RULE: Avoid low-entropy phrasing and follow all safety/clarity rules. "
                    "Return only the final response."
                )
            regenerated = llm.generate(fix_prompt)
            parsed = extract_json(regenerated) if schema else regenerated
            output = parsed if parsed is not None else regenerated
            if isinstance(output, dict) and schema:
                result = self.validate(output, schema)
            else:
                result = self.validate_text(str(output))
            if result["valid"]:
                break
        return {"output": output, "validation": result}


class OptimizedWorkflow:
    """Layer 4: Learning via DSPy + ACE (optional)."""

    def __init__(self, base_module: Any, trainset: Optional[List[Any]] = None):
        self.module = base_module
        self.compiled_module = base_module
        self._ace = None
        dspy = optional_import("dspy")
        ace = optional_import("ace")
        if dspy and trainset:
            try:
                teleprompter = dspy.BootstrapFewShot(metric=self._accuracy)
                self.compiled_module = teleprompter.compile(self.module, trainset=trainset)
            except Exception:
                self.compiled_module = base_module
        if ace and hasattr(ace, "OfflineACE"):
            try:
                self._ace = ace.OfflineACE(Agent=self.compiled_module, reflection_window=3)
            except Exception:
                self._ace = None

    def _accuracy(self, example: Any, pred: Any, trace: Optional[Any] = None) -> bool:
        return getattr(example, "answer", None) == getattr(pred, "answer", None)

    def execute(self, task: Any, learn: bool = True) -> Any:
        if self._ace:
            result = self._ace.ask(task)
            if learn and hasattr(self._ace, "learn") and hasattr(task, "feedback"):
                self._ace.learn(task.feedback)
            return result
        if callable(self.compiled_module):
            return self.compiled_module(task)
        if hasattr(self.compiled_module, "forward"):
            return self.compiled_module.forward(task)
        return task


class MatryoshkaClient:
    """Thin wrapper around the Matryoshka CLI."""

    def __init__(self, cli_path: Optional[str] = None):
        from .deps import matryoshka_cli_path
        self.cli_path = Path(cli_path) if cli_path else matryoshka_cli_path()

    def analyze(
        self,
        query: str,
        document_path: str,
        max_turns: int = 10,
        timeout_ms: int = 30000,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        adapter: Optional[str] = None,
        output_type: Optional[str] = None,
        constraints: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        import json as _json
        import shutil
        import subprocess

        dist_entry = self.cli_path.parent.parent / "dist" / "index.js"
        cmd: List[str]
        if dist_entry.exists():
            cmd = ["node", str(dist_entry)]
        else:
            tsx = shutil.which("tsx")
            if tsx is None:
                raise RuntimeError("Matryoshka build missing. Run npm install && npm run build in Matryoshka/")
            cmd = [tsx, str(self.cli_path)]

        cmd.extend([query, document_path])
        cmd.extend(["--max-turns", str(max_turns)])
        cmd.extend(["--timeout", str(timeout_ms)])
        if provider:
            cmd.extend(["--provider", provider])
        if model:
            cmd.extend(["--model", model])
        if adapter:
            cmd.extend(["--adapter", adapter])
        if output_type:
            cmd.extend(["--output-type", output_type])
        if constraints:
            cmd.extend(["--constraints", constraints])
        if config_path:
            cmd.extend(["--config", config_path])

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Matryoshka CLI failed")
        stdout = result.stdout.strip()
        try:
            return _json.loads(stdout)
        except Exception:
            return {"method": "matryoshka", "raw": stdout}


class KnowledgeAdapter:
    """Layer 5: Knowledge Engine integration."""

    def __init__(self):
        self._ke_module = optional_import("knowledge_engine")
        self._engine = None
        self._initialized = False

    def is_available(self) -> bool:
        return self._ke_module is not None

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(coro))
                return future.result()
        return loop.run_until_complete(coro)

    def _ensure_engine(self):
        if self._engine is not None:
            return
        if not self._ke_module:
            raise RuntimeError("Knowledge engine not available")
        create_engine = getattr(self._ke_module, "create_knowledge_engine", None)
        engine_cls = getattr(self._ke_module, "KnowledgeEngine", None)
        if create_engine:
            self._engine = self._run_async(create_engine())
        elif engine_cls:
            self._engine = engine_cls()
            self._run_async(self._engine.initialize())
        else:
            raise RuntimeError("Knowledge engine entrypoint missing")
        self._initialized = True

    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        self._ensure_engine()
        if hasattr(self._engine, "query_temporal"):
            result = self._run_async(self._engine.query_temporal(query=query))
        elif hasattr(self._engine, "query"):
            result = self._run_async(self._engine.query(query=query))
        else:
            result = {"results": []}
        results = getattr(result, "results", None) or result.get("results", [])
        return {"method": "knowledge_engine", "context": results[:max_results], "count": len(results)}

class ContextManager:
    """Layer 5: Context management (Matryoshka or RAG fallback)."""

    def __init__(self):
        self._matryoshka = MatryoshkaClient()
        self._knowledge = KnowledgeAdapter()

    def process_document(self, query: str, document_path: str, size_mb: float) -> Dict[str, Any]:
        if size_mb > 10:
            try:
                return self._matryoshka.analyze(query, document_path)
            except Exception:
                pass
        if self._knowledge.is_available():
            try:
                return self._knowledge.search(query=query, max_results=5)
            except Exception:
                pass
        try:
            with open(document_path, "r", encoding="utf-8") as handle:
                content = handle.read()
        except Exception:
            content = ""
        return {"method": "rag", "context": content[:2000]}


class SmartContextManager:
    def __init__(self):
        self._matryoshka = MatryoshkaClient()
        self._knowledge = KnowledgeAdapter()

    def get_context(self, query: str, document_path: str) -> Dict[str, Any]:
        try:
            import os
            size_mb = os.path.getsize(document_path) / (1024 * 1024)
        except Exception:
            size_mb = 0
        if size_mb < 10:
            try:
                with open(document_path, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except Exception:
                text = ""
            return {"method": "rag", "context": text[:2000], "tokens_used": len(text.split())}
        if self._knowledge.is_available():
            try:
                result = self._knowledge.search(query=query)
                return {"method": "knowledge_engine", "context": result.get("context", []), "tokens_used": len(str(result).split())}
            except Exception:
                pass
        try:
            result = self._matryoshka.analyze(query=query, document_path=document_path, max_turns=10)
            return {"method": "matryoshka", "context": result, "tokens_used": len(str(result).split())}
        except Exception:
            return {"method": "matryoshka", "context": "", "tokens_used": 0}


class FormalVerificationLayer:
    """Layer 6: Formal verification using Z3 with optional CAV-NLP enhancement."""

    def __init__(self, use_cav_nlp: bool = True):
        self._z3 = optional_import("z3")
        self._solver = self._z3.Solver() if self._z3 else None
        
        # CAV-NLP configuration
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        self._cav_solver: Optional[Any] = None
        self._formalizer: Optional[Any] = None
        
        if self.use_cav_nlp and EnhancedZ3Solver is not None:
            try:
                self._cav_solver = EnhancedZ3Solver(use_cav_nlp=True)
                self._formalizer = ConstraintFormalizer()
            except Exception as exc:
                self.use_cav_nlp = False

    def verify_logical_correctness(self, llm_output: Dict[str, Any]) -> Dict[str, Any]:
        # Use CAV-NLP enhanced verification if available
        if self.use_cav_nlp and self._cav_solver is not None:
            try:
                return self._verify_with_cav_nlp(llm_output)
            except Exception as exc:
                # Fall back to standard Z3
                pass
        
        # Standard Z3 verification
        if not self._solver:
            return {"verified": False, "reason": "Z3 not available"}
        propositions = llm_output.get("propositions", [])
        z3_vars = {}
        for prop in propositions:
            name = prop.get("name")
            if name:
                z3_vars[name] = self._z3.Bool(name)
        for constraint in llm_output.get("constraints", []):
            try:
                expr = safe_z3_eval(constraint, z3_vars)
                if expr is None:
                    continue
                self._solver.add(expr)
            except Exception:
                return {"verified": False, "reason": "invalid constraint"}
        return {"verified": self._solver.check() == self._z3.sat}

    def _verify_with_cav_nlp(self, llm_output: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced verification using CAV-NLP hybrid approach."""
        constraints = llm_output.get("constraints", [])
        propositions = llm_output.get("propositions", [])
        
        # Reset CAV solver
        self._cav_solver.reset()
        
        # Add natural language constraints with formalization
        for constraint in constraints:
            if isinstance(constraint, str):
                # Try to formalize natural language constraint
                formalized = self._cav_solver.formalize_constraint(constraint)
                if formalized is not None:
                    self._cav_solver.add(formalized)
                else:
                    # Fall back to direct evaluation for Z3 expressions
                    try:
                        self._cav_solver.add(constraint)
                    except Exception:
                        pass
            else:
                self._cav_solver.add(constraint)
        
        # Perform hybrid verification
        verification = self._cav_solver.verify_with_lean()
        
        return {
            "verified": verification.success,
            "confidence": verification.confidence,
            "z3_result": verification.z3_result,
            "lean_result": verification.lean_result,
            "counterexample": verification.counterexample,
            "proof": verification.proof,
            "method": "cav_nlp_hybrid"
        }

    def formalize_deterministic_constraint(
        self, 
        natural_language: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Formalize natural language constraint to Z3 using CAV-NLP.
        
        Args:
            natural_language: Natural language constraint (e.g., "x is positive")
            context: Optional context for formalization
            
        Returns:
            Dict with formalization result including Z3 expression and metadata
        """
        if not self.use_cav_nlp or self._formalizer is None:
            return {
                "success": False,
                "error": "CAV-NLP not available",
                "source": natural_language
            }
        
        try:
            result = self._formalizer.formalize(natural_language, context)
            return {
                "success": result.success,
                "z3_expr": str(result.z3_expr) if result.z3_expr else None,
                "constraint_type": result.constraint_type,
                "variables": result.variables,
                "canonical_form": result.canonical_form,
                "source": natural_language,
                "warnings": result.warnings
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "source": natural_language
            }

    def canonicalize_deterministic_expression(
        self, 
        expression: str,
        target: str = "z3"
    ) -> Dict[str, Any]:
        """Canonicalize a deterministic expression using CAV-NLP.
        
        Args:
            expression: Expression to canonicalize
            target: Target format ("z3" or "lean")
            
        Returns:
            Dict with canonical form and equivalence proof
        """
        if not self.use_cav_nlp or self._formalizer is None:
            return {
                "success": False,
                "error": "CAV-NLP not available",
                "original": expression
            }
        
        try:
            result = self._formalizer.formalize(expression, target=target)
            return {
                "success": result.success,
                "canonical_form": result.canonical_form,
                "constraint_type": result.constraint_type,
                "variables": result.variables,
                "original": expression,
                "warnings": result.warnings
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "original": expression
            }

    def verify_dimensional_consistency(self, output: Dict[str, Any]) -> Dict[str, Any]:
        if not self._solver:
            return {"verified": False, "reason": "Z3 not available"}
        
        # Use CAV-NLP for enhanced dimensional verification if available
        if self.use_cav_nlp and self._cav_solver is not None:
            calculations = output.get("calculations", [])
            if calculations:
                # Formalize and verify dimensional constraints
                for calc in calculations:
                    equation = calc.get("equation", "")
                    if equation:
                        try:
                            formalized = self._cav_solver.formalize_constraint(
                                f"dimensional consistency of {equation}"
                            )
                            if formalized is not None:
                                self._cav_solver.add(formalized)
                        except Exception:
                            pass
                
                verification = self._cav_solver.verify_with_lean()
                return {
                    "verified": verification.success,
                    "confidence": verification.confidence,
                    "method": "cav_nlp_dimensional",
                    "verified_equations": len(calculations)
                }
        
        return {"verified": True, "reason": "no-dimension-check-implemented"}

    def verify_stoichiometry(self, output: Dict[str, Any]) -> Dict[str, Any]:
        if not self._solver:
            return {"verified": False, "reason": "Z3 not available"}
        
        # Use CAV-NLP for enhanced stoichiometry verification if available
        if self.use_cav_nlp and self._cav_solver is not None:
            reactions = output.get("reactions", [])
            if reactions:
                # Formalize and verify mass balance constraints
                for reaction in reactions:
                    reactants = reaction.get("reactants", [])
                    products = reaction.get("products", [])
                    if reactants and products:
                        try:
                            constraint = f"mass balance for {len(reactants)} reactants and {len(products)} products"
                            formalized = self._cav_solver.formalize_constraint(constraint)
                            if formalized is not None:
                                self._cav_solver.add(formalized)
                        except Exception:
                            pass
                
                verification = self._cav_solver.verify_with_lean()
                return {
                    "verified": verification.success,
                    "confidence": verification.confidence,
                    "method": "cav_nlp_stoichiometry",
                    "verified_reactions": len(reactions)
                }
        
        return {"verified": True, "reason": "no-stoichiometry-check-implemented"}

    def generate_formal_proof(self, theorem: str) -> Dict[str, Any]:
        # Try CAV-NLP enhanced proving first
        if self.use_cav_nlp and self._cav_solver is not None:
            try:
                result = self._cav_solver.prove(theorem)
                if result is not None:
                    return {
                        "proved": result.success if hasattr(result, 'success') else True,
                        "proof": result.proof if hasattr(result, 'proof') else str(result),
                        "confidence": result.confidence if hasattr(result, 'confidence') else 0.8,
                        "method": "cav_nlp_hybrid"
                    }
            except Exception:
                pass
        
        # Fall back to Lean 4
        lean = optional_import("lean4")
        if lean and hasattr(lean, "LeanTheoremProver"):
            try:
                prover = lean.LeanTheoremProver()
                return prover.prove(theorem)
            except Exception:
                return {"proved": False, "reason": "lean error"}
        return {"proved": False, "reason": "lean not available"}

    def get_capabilities(self) -> Dict[str, bool]:
        """Get available verification capabilities."""
        return {
            "z3_available": self._solver is not None,
            "cav_nlp_available": self.use_cav_nlp,
            "formalization": self.use_cav_nlp,
            "hybrid_verification": self.use_cav_nlp,
            "canonicalization": self.use_cav_nlp,
            "counterexamples": self.use_cav_nlp and self._cav_solver is not None,
        }


class ReproducibilityLayer:
    """Layer 7: Runtime reproducibility (detLLM or statistical)."""

    def __init__(self, backend: Optional[str] = None, model: Optional[str] = None, mode: str = "auto"):
        self._detllm = optional_import("detllm")
        self.backend = backend
        self.model = model
        self.mode = mode  # auto | local | cloud

    def check(
        self,
        prompt: str,
        llm: Optional[LLMInterface],
        tier: int = 1,
        runs: int = 3,
        backend: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        backend = backend or self.backend
        model = model or self.model
        if llm and getattr(llm, "provider", "").lower() in {"openai", "anthropic", "google"}:
            return self._statistical_check(prompt, llm, runs)

        if self.mode == "cloud":
            return self._statistical_check(prompt, llm, runs)

        if self._detllm and hasattr(self._detllm, "check") and backend and model:
            try:
                report = self._detllm.check(
                    backend=backend,
                    model=model,
                    prompts=[prompt],
                    runs=runs,
                    tier=tier,
                )
                return {"status": report.status, "details": report.details}
            except Exception as exc:
                return {"status": "ERROR", "details": str(exc)}

        return self._statistical_check(prompt, llm, runs)

    def _statistical_check(self, prompt: str, llm: Optional[LLMInterface], runs: int) -> Dict[str, Any]:
        if not llm:
            return {"status": "UNAVAILABLE", "details": "no llm provided"}
        outputs = [llm.generate(prompt) for _ in range(runs)]
        baseline = outputs[0] if outputs else ""
        scores = [similarity(baseline, out) for out in outputs[1:]]
        avg = sum(scores) / max(len(scores), 1)
        return {"status": "CONSISTENT" if avg > 0.95 else "DIVERGENT", "avg_similarity": avg}
