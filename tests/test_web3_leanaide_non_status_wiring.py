import asyncio

import pytest


REQUIRED_WEB3_STATUS_KEYS = [
    "web3_formal_available",
    "web3_formal_verification_available",
    "web3_formal_tools",
    "formal_capabilities",
    "audit_exploit_verification_available",
]


def _as_dict(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _assert_web3_status_schema(payload):
    for key in REQUIRED_WEB3_STATUS_KEYS:
        assert key in payload


def test_mcp_tool_decorator_injects_web3_formal_schema():
    tools = pytest.importorskip("leanaide_mcp_tools")

    @tools.mcp_tool("tmp_web3_status_injection_test")
    def _tmp_tool():
        return {"success": True}

    result = tools.get_mcp_tool("tmp_web3_status_injection_test")()
    _assert_web3_status_schema(result)


def test_leanaide_result_to_dict_exposes_web3_formal_schema():
    integration_mod = pytest.importorskip("knowledge_engine.integrations.leanaide_integration")
    result = integration_mod.LeanAideResult(
        success=True,
        verified=True,
        proof="by trivial",
        theorem="theorem t : True",
        reasoning_trace="ok",
        metadata={},
    )
    payload = result.to_dict()
    _assert_web3_status_schema(payload)


def test_rag_proof_result_to_dict_exposes_web3_formal_schema():
    rag_mod = pytest.importorskip("knowledge_engine.integrations.leanaide_ragbits_integration")
    result = rag_mod.RAGProofResult(
        success=True,
        theorem_name="t",
        informal_statement="True",
        generated_proof="by trivial",
        retrieved_proofs=[],
        verification_status=rag_mod.VerificationStatus.VERIFIED,
        confidence_score=1.0,
        processing_time_ms=1.0,
    )
    payload = result.to_dict()
    _assert_web3_status_schema(payload)


def test_leanaide_bridge_verify_payload_exposes_web3_formal_schema():
    bridge_mod = pytest.importorskip("openevolve.leanaide_cav_nlp_bridge")
    bridge = bridge_mod.LeanAideCAVNLPBridge(use_cav_nlp=False, use_unified_service=False)
    payload = asyncio.run(bridge.verify("theorem t : True := by trivial"))
    _assert_web3_status_schema(payload)


def test_leanaide_verify_endpoint_exposes_web3_formal_schema(monkeypatch):
    routes = pytest.importorskip("leanaide_api_routes")
    monkeypatch.setattr(routes, "LEANAIDE_AVAILABLE", True, raising=False)

    async def _fake_verify(*args, **kwargs):
        return {"success": True, "confidence": 0.9, "metadata": {"mocked": True}}

    monkeypatch.setattr(routes, "leanaide_verify_solution_async", _fake_verify)
    request = routes.LeanAideVerifyRequest(code="theorem t : True := by trivial", timeout=1)
    response = asyncio.run(routes.leanaide_verify(request))
    payload = _as_dict(response)
    _assert_web3_status_schema(payload)
    assert "web3_formal_status" in payload["metadata"]


def test_leanaide_prove_endpoint_exposes_web3_formal_schema(monkeypatch):
    routes = pytest.importorskip("leanaide_api_routes")
    monkeypatch.setattr(routes, "LEANAIDE_AVAILABLE", True, raising=False)

    async def _fake_translate(*args, **kwargs):
        return {"success": True, "lean_code": "theorem t : True := by trivial", "name": "t"}

    async def _fake_generate(*args, **kwargs):
        return {"success": True, "proof": "by trivial", "confidence": 0.95}

    monkeypatch.setattr(routes, "leanaide_translate_theorem_async", _fake_translate)
    monkeypatch.setattr(routes, "leanaide_generate_proof_async", _fake_generate)

    request = routes.LeanAideProveRequest(theorem_text="True", theorem_name="t", timeout=1)
    response = asyncio.run(routes.leanaide_prove(request))
    payload = _as_dict(response)
    _assert_web3_status_schema(payload)
    assert "web3_formal_status" in payload["metadata"]


def test_leanaide_translate_endpoint_exposes_web3_formal_schema(monkeypatch):
    routes = pytest.importorskip("leanaide_api_routes")
    monkeypatch.setattr(routes, "LEANAIDE_AVAILABLE", True, raising=False)

    async def _fake_translate(*args, **kwargs):
        return {
            "success": True,
            "name": "t",
            "lean_code": "theorem t : True := by trivial",
            "metadata": {"mocked": True},
        }

    monkeypatch.setattr(routes, "leanaide_translate_theorem_async", _fake_translate)
    request = routes.LeanAideTranslateRequest(theorem_text="True", name="t", timeout=1)
    response = asyncio.run(routes.leanaide_translate(request))
    payload = _as_dict(response)
    _assert_web3_status_schema(payload)
    assert "web3_formal_status" in payload["metadata"]


def test_leanaide_quality_gate_endpoint_exposes_web3_formal_schema(monkeypatch):
    routes = pytest.importorskip("leanaide_api_routes")
    monkeypatch.setattr(routes, "QUALITY_GATE_AVAILABLE", True, raising=False)

    class _Result:
        verification_passed = True
        confidence_score = 0.99
        is_mathematical = True
        errors = []

    class _Verifier:
        async def verify_mathematical_correctness(self, *_args, **_kwargs):
            return _Result()

    monkeypatch.setattr(routes, "get_quality_gate_verifier", lambda: _Verifier())
    request = routes.LeanAideQualityGateRequest(solution_content="proof", confidence_threshold=0.8)
    response = asyncio.run(routes.leanaide_quality_gate(request))
    payload = _as_dict(response)
    _assert_web3_status_schema(payload)


def test_rag_endpoints_expose_web3_formal_schema(monkeypatch):
    routes = pytest.importorskip("leanaide_api_routes")
    monkeypatch.setattr(routes, "RAGBITS_INTEGRATION_AVAILABLE", True, raising=False)

    class _Proof:
        def to_dict(self):
            return {"id": "p1"}

    class _RagResult:
        def to_dict(self):
            return {"success": True, "theorem_name": "t"}

    class _Integration:
        async def retrieve_similar_proofs(self, *_args, **_kwargs):
            return [_Proof()]

        async def generate_proof_with_retrieval(self, *_args, **_kwargs):
            return _RagResult()

    monkeypatch.setattr(routes, "get_ragbits_integration", lambda: _Integration())

    retrieve_payload = asyncio.run(routes.leanaide_rag_retrieve("t", top_k=1))
    _assert_web3_status_schema(retrieve_payload)
    prove_payload = asyncio.run(routes.leanaide_rag_prove("t", theorem_name="t"))
    _assert_web3_status_schema(prove_payload)


def test_cav_nlp_endpoints_expose_web3_formal_schema(monkeypatch):
    routes = pytest.importorskip("leanaide_api_routes")
    monkeypatch.setattr(routes, "CAV_NLP_AVAILABLE", True, raising=False)

    class _MathService:
        async def formalize_async(self, **_kwargs):
            return {
                "success": True,
                "lean_code": "theorem t : True := by trivial",
                "name": "t",
                "confidence": 0.9,
                "constraints_used": [],
                "verification_status": "pending",
                "metadata": {},
            }

        async def analyze_semantics_async(self, **_kwargs):
            return {"semantic_score": 0.95}

    class _Solver:
        async def check_constraints_async(self, **_kwargs):
            return {"satisfiable": True}

    monkeypatch.setattr(routes, "get_math_service", lambda: _MathService())
    monkeypatch.setattr(routes, "get_enhanced_solver", lambda: _Solver())

    formalize_request = routes.CAVNLPFormalizeRequest(natural_language="True")
    formalize_payload = _as_dict(asyncio.run(routes.cav_nlp_formalize(formalize_request)))
    _assert_web3_status_schema(formalize_payload)

    verify_request = routes.CAVNLPVerifyRequest(lean_code="theorem t : True := by trivial")
    verify_payload = _as_dict(asyncio.run(routes.cav_nlp_verify(verify_request)))
    _assert_web3_status_schema(verify_payload)
