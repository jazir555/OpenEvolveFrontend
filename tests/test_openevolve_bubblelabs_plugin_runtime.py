import asyncio
from types import SimpleNamespace

from bubblelabs_plugin_system import PluginState
from openevolve_bubblelabs_plugin import OpenEvolveBubbleLabsPlugin


def test_run_sync_supports_keyword_arguments():
    plugin = OpenEvolveBubbleLabsPlugin({})

    async def _run():
        return await plugin._run_sync(lambda *, value: value + 1, value=41)

    assert asyncio.run(_run()) == 42


def test_cancel_all_workflows_handles_dict_instances():
    plugin = OpenEvolveBubbleLabsPlugin({})
    plugin._integration = SimpleNamespace(
        list_workflow_instances=lambda: [
            {"instance_id": "wf-running", "status": "running"},
            {"instance_id": "wf-complete", "status": "completed"},
        ]
    )
    calls = []

    async def fake_control(instance_id, action):
        calls.append((instance_id, action))
        return {"status": "ok"}

    plugin.control_workflow = fake_control
    asyncio.run(plugin._cancel_all_workflows())

    assert calls == [("wf-running", "cancel")]


def test_stop_handles_cancelled_cleanup_task_without_error():
    plugin = OpenEvolveBubbleLabsPlugin({})
    plugin._integration = SimpleNamespace(list_workflow_instances=lambda: [])

    async def _run():
        plugin._cleanup_task = asyncio.create_task(asyncio.sleep(30))
        await plugin.stop()
        assert plugin._cleanup_task is None
        assert plugin._status.state == PluginState.STOPPED

    asyncio.run(_run())
