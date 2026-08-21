"""
Unit tests for Raft cluster membership changes and heartbeat-based
failure detection in knowledge_engine.distributed_coordination.RaftNode.

Uses a deterministic, injectable fake clock so the tests are fast and
free of wall-clock timing flakiness.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from distributed_coordination import (
    NodeState,
    RaftNode,
    MembershipChangeError,
)


class FakeClock:
    """Injectable monotonic clock: returns a controllable float time."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_node(
    node_id,
    peers,
    *,
    failure_timeout=1.0,
    auto_remove_failed=False,
    clock=None,
):
    data_dir = tempfile.mkdtemp(prefix=f"raft_test_{node_id}_")
    return RaftNode(
        node_id=node_id,
        address="127.0.0.1",
        port=9000,
        peers=peers,
        data_dir=data_dir,
        failure_timeout=failure_timeout,
        auto_remove_failed=auto_remove_failed,
        clock=clock or FakeClock(),
    )


def make_leader(node_id="n1", peer_ids=("n2", "n3")):
    clock = FakeClock()
    peers = [(pid, "127.0.0.1", 9000 + i) for i, pid in enumerate(peer_ids, start=1)]
    node = make_node(node_id, peers, clock=clock)
    # Force leader state deterministically (no async election races).
    node.state = NodeState.LEADER
    node.current_leader = node.node_id
    for pid in node._active_peer_ids():
        node.volatile_state.next_index[pid] = 1
        node.volatile_state.match_index[pid] = 0
    return node, clock


# ----------------------------------------------------------------------
# Membership changes
# ----------------------------------------------------------------------
def test_add_member_updates_peer_set_and_config():
    node, _ = make_leader()
    assert len(node.peers) == 2
    assert "n4" not in node.get_member_ids()

    node.add_member("n4", "127.0.0.1", 9004)

    assert "n4" in node.get_member_ids()
    assert node.is_member("n4")
    assert len(node.peers) == 3
    assert node.peers["n4"] == ("127.0.0.1", 9004)
    assert node.get_cluster_config()["phase"] == "stable"


def test_remove_member_updates_peer_set_and_config():
    node, _ = make_leader()
    assert "n2" in node.get_member_ids()

    node.remove_member("n2")

    assert "n2" not in node.get_member_ids()
    assert not node.is_member("n2")
    assert "n2" not in node.peers
    assert len(node.peers) == 1


def test_add_existing_member_raises():
    node, _ = make_leader()
    with pytest.raises(MembershipChangeError):
        node.add_member("n2", "127.0.0.1", 9001)


def test_remove_unknown_member_raises():
    node, _ = make_leader()
    with pytest.raises(MembershipChangeError):
        node.remove_member("ghost")


def test_membership_change_requires_leader():
    node = make_node(
        "n1", [("n2", "127.0.0.1", 9001), ("n3", "127.0.0.1", 9002)]
    )
    assert node.state == NodeState.FOLLOWER
    with pytest.raises(MembershipChangeError):
        node.add_member("n4", "127.0.0.1", 9004)
    with pytest.raises(MembershipChangeError):
        node.remove_member("n2")


def test_follower_apply_membership_update():
    node = make_node(
        "n1", [("n2", "127.0.0.1", 9001), ("n3", "127.0.0.1", 9002)]
    )
    node.receive_membership_update({"n2": ("10.0.0.2", 7002), "n4": ("10.0.0.4", 7004)})
    assert node.is_member("n2")
    assert node.is_member("n4")
    assert not node.is_member("n3")
    assert node.peers["n4"] == ("10.0.0.4", 7004)


# ----------------------------------------------------------------------
# Failure detection
# ----------------------------------------------------------------------
def test_heartbeat_timeout_marks_peer_down():
    clock = FakeClock()
    node = make_node(
        "n1", [("n2", "127.0.0.1", 9001)], clock=clock, failure_timeout=1.0
    )
    assert node.is_alive("n2")

    clock.advance(0.5)
    node.tick()
    assert node.is_alive("n2")  # still within timeout

    clock.advance(0.6)  # now 1.1s since last heartbeat
    down = node.tick()
    assert "n2" in down
    assert node.get_member_status("n2") == "down"
    assert not node.is_alive("n2")


def test_record_heartbeat_resets_failure():
    clock = FakeClock()
    node = make_node(
        "n1", [("n2", "127.0.0.1", 9001)], clock=clock, failure_timeout=1.0
    )
    clock.advance(2.0)
    node.tick()
    assert not node.is_alive("n2")

    node.record_heartbeat("n2")
    assert node.is_alive("n2")
    assert node.get_member_status("n2") == "alive"


def test_leader_auto_removes_downed_peer():
    node, clock = make_leader(peer_ids=("n2", "n3"))
    node._auto_remove_failed = True

    # Only n2's heartbeat lapses; keep n3 alive.
    clock.advance(2.0)
    node.record_heartbeat("n3")
    node.tick()

    assert "n2" not in node.get_member_ids()
    assert "n3" in node.get_member_ids()


def test_follower_triggers_relection_when_leader_down():
    clock = FakeClock()
    node = make_node(
        "n1", [("n2", "127.0.0.1", 9001)], clock=clock, failure_timeout=1.0
    )
    node.state = NodeState.FOLLOWER
    node.current_leader = "n2"  # our leader is a peer

    called = {"v": False}

    async def fake_start_election():
        called["v"] = True

    node._start_election = fake_start_election

    clock.advance(2.0)

    async def _run():
        node.tick()  # runs inside the loop so the election task can be scheduled
        await asyncio.sleep(0)  # let the scheduled election task execute

    asyncio.run(_run())
    assert called["v"] is True


def test_leader_handles_membership_change_end_to_end():
    node, _ = make_leader(peer_ids=("n2", "n3"))
    # Leader adds, then removes, a member; peer set reflects both changes.
    node.add_member("n4", "127.0.0.1", 9004)
    assert set(node.get_member_ids()) == {"n1", "n2", "n3", "n4"}
    node.remove_member("n4")
    assert set(node.get_member_ids()) == {"n1", "n2", "n3"}
    # Removal of a current member also works.
    node.remove_member("n3")
    assert set(node.get_member_ids()) == {"n1", "n2"}
