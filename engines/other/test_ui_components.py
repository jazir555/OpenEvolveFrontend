import os
import sys

# Flat-style: insert engines/other + repo root into sys.path at top.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ui_components


def _import_orchestrator():
    """Import openevolve_orchestrator, fixing a pre-existing sys.path shadow that
    makes the top-level ``utils`` package unimportable in some environments.

    Returns the module, or raises ImportError if it still cannot be imported.
    """
    import os
    import sys
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
    sys.path.insert(0, os.path.join(root, "engines", "other"))
    sys.path.insert(0, root)
    # Drop shadowing entries (e.g. a bare ``rese`` dir) that make ``utils`` resolve
    # as a non-package.
    sys.path = [
        p for p in sys.path
        if not (os.path.basename(p.rstrip("/\\")) == "rese"
                or p.replace("\\", "/").endswith("api/gateway"))
    ]
    # Clear any badly-cached ``utils`` namespace so the real package wins.
    for mod in list(sys.modules):
        if mod == "utils" or mod.startswith("utils."):
            del sys.modules[mod]
    import openevolve_orchestrator as orch
    return orch


def test_render_analytics_dashboard_nonempty():
    fake_state = {
        "workflow_id": "wf_1",
        "status": "completed",
        "current_stage": "Analysis",
        "start_time": 1.0,
        "end_time": 3.5,
    }
    fake_metrics = {
        "steps_completed": 5,
        "steps_failed": 1,
        "red_flags": ["flag a"],
        "votes": {"yes": 3, "no": 1},
        "teams_performance": {"Blue": 0.9},
    }
    out = ui_components.render_analytics_dashboard(fake_state, fake_metrics)
    assert isinstance(out, str) and out.strip(), "analytics dashboard must return non-empty string"
    assert "Analytics Dashboard" in out


def test_render_knowledge_base_interface_nonempty():
    out = ui_components.render_knowledge_base_interface(None)
    assert isinstance(out, str) and out.strip(), "knowledge base interface must return non-empty string"


def test_render_dependency_graph_cycle():
    subs = [
        {"id": "a", "dependencies": ["c"]},
        {"id": "b", "dependencies": ["a"]},
        {"id": "c", "dependencies": ["b"]},
    ]
    out = ui_components.render_dependency_graph(subs)
    assert isinstance(out, str) and out.strip()
    assert "mermaid" in out
    assert "WARNING" in out and "circular" in out

    subs2 = [
        {"id": "a", "dependencies": []},
        {"id": "b", "dependencies": ["a"]},
    ]
    out2 = ui_components.render_dependency_graph(subs2)
    assert "WARNING" not in out2


def test_render_manual_review_panel_smoke():
    assert callable(ui_components.render_manual_review_panel)


def test_openevolve_orchestrator_imports_and_auto_approval():
    orch = _import_orchestrator()

    # monkeypatch run_sovereign_workflow to prove the auto-approval path is reachable
    called = {}

    def fake_run(**kwargs):
        called["ran"] = True
        return None

    orch.run_sovereign_workflow = fake_run

    class FakeState:
        auto_approval = True
        batch_size = 4

    assert orch._resolve_sovereign_auto_approval(FakeState()) is True
    assert orch._resolve_sovereign_batch_size(FakeState()) == 4

    class FakeStateOff:
        auto_approval = False

    assert orch._resolve_sovereign_auto_approval(FakeStateOff()) is False
    assert orch._resolve_sovereign_auto_approval(None) is False
    assert orch._resolve_sovereign_batch_size(None, default=1) == 1
    # ensure the monkeypatched entry point is callable (no breakage)
    orch.run_sovereign_workflow()
    assert called.get("ran") is True
