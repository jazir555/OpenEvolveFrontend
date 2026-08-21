"""
End-to-end tests for the OpenEvolve gRPC NodeRegistry.

Starts a real gRPC server on an OS-assigned port in a background thread and
drives it through the real Python client. No network access, no mocks, and no
`grpcio-health-checking`/`grpcio-reflection` required.

Run with:
    python -m pytest test_grpc_e2e.py -v
"""

import asyncio
import threading

import grpc
import pytest

from client import (
    ExecutionProgress,
    ExecutionRequest,
    OpenEvolveGRPCClient,
    GRPCClientConfig,
)
from generated import common_pb2, nodes_pb2
from server import OpenEvolveGRPCServer, ServerConfig


@pytest.fixture(scope="module")
def grpc_server():
    """Start the gRPC server in a background thread on an ephemeral port."""
    config = ServerConfig(
        host="127.0.0.1",
        port=0,                  # ask the OS for a free port
        max_workers=4,
        enable_reflection=False,  # optional dep; not needed for these tests
        use_real_nodes=False,     # local seed nodes keep the test offline/fast
    )
    server = OpenEvolveGRPCServer(config)
    port = server.start(block=False)

    # Nothing else to wait for: start() returns after the port is bound.
    thread = threading.Thread(target=server._wait_for_shutdown, daemon=True)
    thread.start()

    yield server, port

    server.stop(grace_period=1.0)
    thread.join(timeout=5)


@pytest.fixture
def client_factory(grpc_server):
    """Build a connected client, and clean it up afterwards."""
    _, port = grpc_server
    clients = []

    async def _make() -> OpenEvolveGRPCClient:
        client = OpenEvolveGRPCClient(
            GRPCClientConfig(host="127.0.0.1", port=port, connect_timeout_ms=5000)
        )
        await client.connect()
        clients.append(client)
        return client

    yield _make

    async def _close():
        for client in clients:
            await client.close()

    asyncio.run(_close())


def run(coro):
    """Run a coroutine to completion (avoids needing pytest-asyncio)."""
    return asyncio.run(coro)


class TestServerStartup:
    """The server must actually bind and register the application service."""

    def test_server_bound_to_port(self, grpc_server):
        server, port = grpc_server
        assert port > 0
        assert server.bound_port == port

    def test_servicer_has_nodes(self, grpc_server):
        server, _ = grpc_server
        nodes = server.servicer.node_adapter.list_nodes()
        assert len(nodes) > 0, "registry should be seeded with nodes"


class TestNodeRegistryRPCs:
    """Real RPCs over the wire - these all returned UNIMPLEMENTED before."""

    def test_list_nodes_returns_real_data(self, client_factory):
        async def scenario():
            client = await client_factory()
            nodes = await client.list_nodes()
            assert len(nodes) > 0, "ListNodes must not return an empty list"

            by_id = {n.node_id: n for n in nodes}
            assert "decomposition" in by_id
            assert "echo" in by_id

            decomposition = by_id["decomposition"]
            assert decomposition.display_name == "Problem Decomposition"
            assert decomposition.category == "analysis"
            assert decomposition.version
            assert decomposition.capabilities["supports_streaming"] is True
            assert decomposition.parameter_schema["type"] == "object"

        run(scenario())

    def test_list_nodes_is_not_unimplemented(self, grpc_server):
        """Guard against a regression to the commented-out registration."""
        _, port = grpc_server
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            from generated import nodes_pb2_grpc

            stub = nodes_pb2_grpc.NodeRegistryStub(channel)
            response = stub.ListNodes(nodes_pb2.ListNodesRequest(), timeout=5)
            assert response.pagination.total_count == len(response.nodes)
            assert len(response.nodes) > 0

    def test_list_nodes_category_filter(self, client_factory):
        async def scenario():
            client = await client_factory()
            analysis = await client.list_nodes(category="analysis")
            assert analysis, "expected at least one analysis node"
            assert all(n.category == "analysis" for n in analysis)

        run(scenario())

    def test_get_node_schema(self, client_factory):
        async def scenario():
            client = await client_factory()
            info = await client.get_node_schema("decomposition")
            assert info.node_id == "decomposition"
            assert info.node_type == "decomposition"
            assert "max_subproblems" in info.parameter_schema["properties"]

        run(scenario())

    def test_get_node_schema_unknown_node(self, client_factory):
        async def scenario():
            client = await client_factory()
            with pytest.raises(grpc.RpcError) as excinfo:
                # NODE_TYPE_GAUNTLET is a valid enum value but is not registered.
                await client.get_node_schema("gauntlet")
            assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND

        run(scenario())

    def test_execute_node(self, client_factory):
        async def scenario():
            client = await client_factory()
            result = await client.execute_node(
                ExecutionRequest(
                    node_type="decomposition",
                    inputs={"problem_statement": "Build a compiler and test it"},
                    config={"max_subproblems": 2},
                )
            )
            assert result.state == "COMPLETED"
            assert result.execution_id
            assert result.result["count"] == 2
            assert len(result.result["subproblems"]) == 2
            assert result.metrics["artifact_count"] == 0

        run(scenario())

    def test_execute_node_echoes_inputs(self, client_factory):
        async def scenario():
            client = await client_factory()
            result = await client.execute_node(
                ExecutionRequest(node_type="echo", inputs={"hello": "world"})
            )
            assert result.state == "COMPLETED"
            assert result.result["echo"] == {"hello": "world"}

        run(scenario())

    def test_execute_node_validation_error_is_reported(self, client_factory):
        async def scenario():
            client = await client_factory()
            # `decomposition` requires problem_statement.
            result = await client.execute_node(
                ExecutionRequest(node_type="decomposition", inputs={})
            )
            assert result.state == "FAILED"
            assert result.error is not None
            assert "problem_statement" in result.error["message"]

        run(scenario())

    def test_execute_unknown_node_returns_not_found(self, client_factory):
        async def scenario():
            client = await client_factory()
            with pytest.raises(grpc.RpcError) as excinfo:
                await client.execute_node(
                    ExecutionRequest(node_type="does_not_exist", inputs={})
                )
            assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND

        run(scenario())

    def test_execute_node_streaming(self, client_factory):
        async def scenario():
            client = await client_factory()
            updates = []

            def on_progress(progress: ExecutionProgress):
                updates.append(progress)

            result = await client.execute_node_streaming(
                ExecutionRequest(
                    node_type="semantic_search",
                    inputs={"text": "one two three"},
                ),
                on_progress,
            )
            assert result.state == "COMPLETED"
            assert result.result["words"] == 3
            # At minimum the initial 0% and final 100% updates.
            assert updates, "expected progress callbacks"
            assert updates[0].percent == 0
            assert updates[-1].percent == 100

        run(scenario())

    def test_execute_node_streaming_reports_intermediate_progress(self, client_factory):
        """The decomposition node publishes per-subproblem progress."""
        async def scenario():
            client = await client_factory()
            updates = []

            result = await client.execute_node_streaming(
                ExecutionRequest(
                    node_type="decomposition",
                    inputs={"problem_statement": "alpha. beta. gamma. delta."},
                ),
                updates.append,
            )
            assert result.state == "COMPLETED"
            assert result.result["count"] == 4
            messages = [u.message for u in updates]
            assert any("Decomposing subproblem" in m for m in messages), messages
            assert updates[-1].percent == 100

        run(scenario())

    def test_execute_batch(self, client_factory):
        async def scenario():
            client = await client_factory()
            results = await client.execute_batch(
                [
                    ExecutionRequest(node_type="echo", inputs={"i": 1}),
                    ExecutionRequest(node_type="semantic_search", inputs={"text": "a b"}),
                    ExecutionRequest(node_type="decomposition", inputs={}),  # fails
                ],
                parallel=True,
            )
            assert len(results) == 3
            assert results[0].state == "COMPLETED"
            assert results[1].result["words"] == 2
            assert results[2].state == "FAILED"

        run(scenario())

    def test_get_execution_status_after_completion(self, client_factory):
        async def scenario():
            client = await client_factory()
            result = await client.execute_node(
                ExecutionRequest(node_type="echo", inputs={"a": 1})
            )
            status = await client.get_execution_status(result.execution_id)
            assert status.execution_id == result.execution_id
            assert status.state == "COMPLETED"

        run(scenario())

    def test_get_execution_status_unknown(self, client_factory):
        async def scenario():
            client = await client_factory()
            with pytest.raises(grpc.RpcError) as excinfo:
                await client.get_execution_status("nope")
            assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND

        run(scenario())

    def test_cancel_unknown_execution_reports_failure(self, client_factory):
        async def scenario():
            client = await client_factory()
            assert await client.cancel_execution("not-running") is False

        run(scenario())

    def test_health_check(self, client_factory):
        async def scenario():
            client = await client_factory()
            health = await client.check_health()
            assert health["serving"] is True
            assert health["status"] == "SERVING"

        run(scenario())


class TestClientGuards:
    """Client-side preconditions."""

    def test_calls_before_connect_raise(self):
        client = OpenEvolveGRPCClient(GRPCClientConfig(host="127.0.0.1", port=1))
        with pytest.raises(RuntimeError, match="not connected"):
            run(client.list_nodes())

    def test_connect_to_dead_port_raises(self):
        client = OpenEvolveGRPCClient(
            GRPCClientConfig(host="127.0.0.1", port=9, connect_timeout_ms=300)
        )
        with pytest.raises(ConnectionError):
            run(client.connect())


class TestProtoMapping:
    """The string<->enum contract shared by client and server."""

    def test_node_type_round_trip(self):
        import proto_mapping as pm

        assert pm.node_type_to_enum("decomposition") == nodes_pb2.NODE_TYPE_DECOMPOSITION
        assert pm.enum_to_node_type(nodes_pb2.NODE_TYPE_DECOMPOSITION) == "decomposition"
        assert pm.node_type_to_enum("not_a_node") == nodes_pb2.NODE_TYPE_UNSPECIFIED

    def test_execution_state_round_trip(self):
        import proto_mapping as pm

        assert pm.execution_state_name(common_pb2.EXECUTION_STATE_COMPLETED) == "COMPLETED"
        assert pm.execution_state_value("COMPLETED") == common_pb2.EXECUTION_STATE_COMPLETED

    def test_struct_round_trip_handles_exotic_values(self):
        import proto_mapping as pm

        data = {"set": {1, 2}, "nan": float("nan"), "nested": {"a": [1, "b"]}}
        restored = pm.struct_to_dict(pm.dict_to_struct(data))
        assert sorted(restored["set"]) == [1, 2]
        assert restored["nan"] == "nan"
        assert restored["nested"] == {"a": [1, "b"]}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
