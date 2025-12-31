"""
Test Suite for Lean 4 Self-Play System

This module provides comprehensive tests for the LeanAide self-play system,
including unit tests, integration tests, and example usage scenarios.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from leanaide_selfplay import (
    LeanTheorem,
    LeanTactic,
    LeanProof,
    ProofDifficulty,
    ProofStatus,
    LeanProofStrategy,
    LeanProofExperience,
    LeanProofExperienceBuffer,
    Lean4Verifier,
    LeanProofAgent,
    LeanSelfPlayGame,
    LeanSelfPlayEngine
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_theorem():
    """Create a sample theorem for testing"""
    return LeanTheorem(
        id="test_theorem_1",
        statement="∀ n : Nat, n + 0 = n",
        lean_code="theorem test : ∀ n : Nat, n + 0 = n := by",
        difficulty=ProofDifficulty.EASY,
        domain="algebra",
        dependencies=["Nat.add_zero"]
    )


@pytest.fixture
def sample_tactics():
    """Create sample tactics"""
    return [
        LeanTactic(name="intro", args=["n"]),
        LeanTactic(name="rw", args=["Nat.add_zero"]),
        LeanTactic(name="rfl")
    ]


@pytest.fixture
def sample_proof(sample_theorem, sample_tactics):
    """Create a sample proof"""
    return LeanProof(
        theorem_id=sample_theorem.id,
        tactics=sample_tactics,
        lean_code="intro n\nrw [Nat.add_zero]\nrfl",
        status=ProofStatus.VERIFIED,
        confidence=0.9
    )


@pytest.fixture
def sample_strategy():
    """Create a sample proof strategy"""
    return LeanProofStrategy(
        name="test_strategy",
        tactic_sequence=["intro", "rw", "rfl"],
        description="Test strategy for unit tests",
        适用领域=["algebra", "logic"],
        success_rate=0.8
    )


# ============================================================================
# Unit Tests
# ============================================================================

class TestLeanTheorem:
    """Tests for LeanTheorem dataclass"""

    def test_theorem_creation(self, sample_theorem):
        """Test creating a theorem"""
        assert sample_theorem.id == "test_theorem_1"
        assert sample_theorem.difficulty == ProofDifficulty.EASY
        assert sample_theorem.domain == "algebra"

    def test_theorem_to_lean_file(self, sample_theorem):
        """Test converting theorem to Lean file format"""
        lean_file = sample_theorem.to_lean_file()

        assert "import Mathlib" in lean_file
        assert sample_theorem.statement in lean_file
        assert "theorem " + sample_theorem.id in lean_file


class TestLeanProof:
    """Tests for LeanProof dataclass"""

    def test_proof_creation(self, sample_proof, sample_theorem, sample_tactics):
        """Test creating a proof"""
        assert sample_proof.theorem_id == sample_theorem.id
        assert sample_proof.tactic_count == len(sample_tactics)
        assert sample_proof.is_valid is True

    def test_proof_invalid(self):
        """Test invalid proof"""
        proof = LeanProof(
            theorem_id="test",
            tactics=[],
            lean_code="",
            status=ProofStatus.FAILED
        )
        assert proof.is_valid is False

    def test_proof_partial(self):
        """Test partial proof"""
        proof = LeanProof(
            theorem_id="test",
            tactics=[LeanTactic(name="intro")],
            lean_code="intro",
            status=ProofStatus.PARTIAL
        )
        assert proof.is_valid is False


class TestLeanTactic:
    """Tests for LeanTactic dataclass"""

    def test_tactic_with_args(self):
        """Test tactic with arguments"""
        tactic = LeanTactic(name="rw", args=["Nat.add_zero"])
        assert str(tactic) == "rw Nat.add_zero"

    def test_tactic_without_args(self):
        """Test tactic without arguments"""
        tactic = LeanTactic(name="rfl")
        assert str(tactic) == "rfl"


class TestLeanProofStrategy:
    """Tests for LeanProofStrategy dataclass"""

    def test_strategy_creation(self, sample_strategy):
        """Test creating a strategy"""
        assert sample_strategy.name == "test_strategy"
        assert len(sample_strategy.tactic_sequence) == 3
        assert "algebra" in sample_strategy.适用领域


class TestLeanProofExperience:
    """Tests for LeanProofExperience dataclass"""

    def test_experience_creation(self, sample_theorem, sample_proof):
        """Test creating an experience"""
        experience = LeanProofExperience(
            theorem=sample_theorem,
            proof=sample_proof,
            reward=1.0,
            strategy_used="test_strategy",
            value_estimate=0.9,
            policy_output={"test_strategy": 0.8}
        )

        assert experience.theorem.id == sample_theorem.id
        assert experience.reward == 1.0
        assert experience.strategy_used == "test_strategy"

    def test_experience_to_dict(self, sample_theorem, sample_proof):
        """Test converting experience to dictionary"""
        experience = LeanProofExperience(
            theorem=sample_theorem,
            proof=sample_proof,
            reward=0.5,
            strategy_used="test",
            value_estimate=0.7,
            policy_output={}
        )

        exp_dict = experience.to_training_dict()

        assert "theorem" in exp_dict
        assert "proof" in exp_dict
        assert "reward" in exp_dict
        assert exp_dict["reward"] == 0.5


# ============================================================================
# Experience Buffer Tests
# ============================================================================

class TestLeanProofExperienceBuffer:
    """Tests for experience buffer"""

    @pytest.fixture
    def buffer(self):
        """Create a buffer for testing"""
        return LeanProofExperienceBuffer(capacity=100)

    @pytest.fixture
    def sample_experiences(self, sample_theorem, sample_proof):
        """Create sample experiences"""
        experiences = []
        for i in range(10):
            proof = LeanProof(
                theorem_id=sample_theorem.id,
                tactics=sample_proof.tactics,
                lean_code=sample_proof.lean_code,
                status=ProofStatus.VERIFIED if i % 2 == 0 else ProofStatus.FAILED,
                confidence=0.8
            )
            exp = LeanProofExperience(
                theorem=sample_theorem,
                proof=proof,
                reward=1.0 if i % 2 == 0 else 0.0,
                strategy_used="test",
                value_estimate=0.9,
                policy_output={}
            )
            experiences.append(exp)
        return experiences

    def test_buffer_creation(self, buffer):
        """Test buffer creation"""
        assert buffer.capacity == 100
        assert len(buffer.buffer) == 0

    def test_add_experience(self, buffer, sample_theorem, sample_proof):
        """Test adding experience to buffer"""
        exp = LeanProofExperience(
            theorem=sample_theorem,
            proof=sample_proof,
            reward=1.0,
            strategy_used="test",
            value_estimate=0.9,
            policy_output={}
        )
        buffer.add(exp)

        assert len(buffer.buffer) == 1
        assert buffer.add_count == 1

    def test_add_experiences(self, buffer, sample_experiences):
        """Test adding multiple experiences"""
        for exp in sample_experiences:
            buffer.add(exp)

        assert len(buffer.buffer) == 10
        assert buffer.add_count == 10

    def test_sample_empty_buffer(self, buffer):
        """Test sampling from empty buffer"""
        batch = buffer.sample(batch_size=5)
        assert batch == []

    def test_sample_from_buffer(self, buffer, sample_experiences):
        """Test sampling from buffer"""
        for exp in sample_experiences:
            buffer.add(exp)

        batch = buffer.sample(batch_size=5)
        assert len(batch) == 5

    def test_buffer_capacity(self, buffer, sample_theorem, sample_proof):
        """Test buffer capacity limit"""
        # Create more experiences than capacity
        for i in range(150):
            exp = LeanProofExperience(
                theorem=sample_theorem,
                proof=sample_proof,
                reward=float(i),
                strategy_used="test",
                value_estimate=0.9,
                policy_output={}
            )
            buffer.add(exp)

        # Buffer should not exceed capacity
        assert len(buffer.buffer) <= buffer.capacity

    def test_buffer_statistics(self, buffer, sample_experiences):
        """Test buffer statistics"""
        for exp in sample_experiences:
            buffer.add(exp)

        stats = buffer.get_statistics()

        assert "size" in stats
        assert "capacity" in stats
        assert "success_rate" in stats
        assert "avg_reward" in stats
        assert stats["size"] == 10

    def test_buffer_save_load(self, buffer, sample_experiences, tmp_path):
        """Test saving and loading buffer"""
        # Add experiences
        for exp in sample_experiences:
            buffer.add(exp)

        # Save to file
        filepath = tmp_path / "buffer.json"
        buffer.save(str(filepath))

        # Load into new buffer
        new_buffer = LeanProofExperienceBuffer(capacity=100)
        new_buffer.load(str(filepath))

        assert len(new_buffer.buffer) == len(buffer.buffer)

    def test_prioritized_sampling(self, sample_theorem, sample_proof):
        """Test prioritized experience replay"""
        buffer = LeanProofExperienceBuffer(
            capacity=100,
            prioritized=True
        )

        # Add experiences with different rewards
        for i in range(10):
            proof = LeanProof(
                theorem_id=sample_theorem.id,
                tactics=sample_proof.tactics,
                lean_code=sample_proof.lean_code,
                status=ProofStatus.VERIFIED,
                confidence=0.8
            )
            exp = LeanProofExperience(
                theorem=sample_theorem,
                proof=proof,
                reward=float(i),  # Increasing rewards
                strategy_used="test",
                value_estimate=0.9,
                policy_output={}
            )
            buffer.add(exp)

        # Sample and check distribution (higher rewards should be sampled more)
        sampled_rewards = []
        for _ in range(100):
            batch = buffer.sample(batch_size=1)
            if batch:
                sampled_rewards.append(batch[0].reward)

        # Last experiences (higher rewards) should appear more often
        assert len(sampled_rewards) > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestLean4Verifier:
    """Tests for Lean 4 verifier"""

    @pytest.mark.asyncio
    async def test_verifier_creation(self):
        """Test creating verifier"""
        verifier = Lean4Verifier(
            leanaide_url="http://localhost:7654",
            timeout=300
        )
        assert verifier.leanaide_url == "http://localhost:7654"
        await verifier.close()

    @pytest.mark.asyncio
    async def test_construct_lean_file(self, sample_theorem, sample_proof):
        """Test Lean file construction"""
        verifier = Lean4Verifier()
        lean_file = verifier._construct_lean_file(sample_theorem, sample_proof)

        assert "import Mathlib" in lean_file
        assert sample_theorem.statement in lean_file
        await verifier.close()

    @pytest.mark.asyncio
    async def test_format_tactics(self):
        """Test tactic formatting"""
        verifier = Lean4Verifier()
        tactics = [
            LeanTactic(name="intro", args=["n"]),
            LeanTactic(name="rw", args=["Nat.add_zero"]),
            LeanTactic(name="rfl")
        ]

        formatted = verifier._format_tactics(tactics)

        assert "intro n" in formatted
        assert "rw Nat.add_zero" in formatted
        assert "rfl" in formatted
        await verifier.close()


class TestLeanProofAgent:
    """Tests for proof agent"""

    @pytest.fixture
    def agent(self):
        """Create an agent for testing"""
        return LeanProofAgent(
            agent_id="test_agent",
            llm_config={},
            verifier=Lean4Verifier(),
            exploration_rate=0.2
        )

    def test_agent_creation(self, agent):
        """Test agent creation"""
        assert agent.agent_id == "test_agent"
        assert agent.exploration_rate == 0.2
        assert len(agent.known_tactics) > 0
        assert len(agent.known_strategies) > 0

    @pytest.mark.asyncio
    async def test_select_proof_strategy(self, agent, sample_theorem):
        """Test strategy selection"""
        strategy = await agent.select_proof_strategy(
            sample_theorem,
            training=False
        )

        assert isinstance(strategy, LeanProofStrategy)
        assert strategy.name in [s.name for s in agent.known_strategies]

    @pytest.mark.asyncio
    async def test_generate_proof(self, agent, sample_theorem):
        """Test proof generation"""
        strategy = LeanProofStrategy(
            name="test",
            tactic_sequence=["intro", "rfl"],
            description="Test",
            适用_domain=["algebra"]
        )

        proof = await agent.generate_proof(sample_theorem, strategy)

        assert isinstance(proof, LeanProof)
        assert proof.theorem_id == sample_theorem.id
        assert proof.generation_time > 0

    @pytest.mark.asyncio
    async def test_evaluate_proof(self, agent):
        """Test proof evaluation"""
        # Valid proof
        valid_proof = LeanProof(
            theorem_id="test",
            tactics=[LeanTactic(name="rfl")],
            lean_code="rfl",
            status=ProofStatus.VERIFIED,
            confidence=0.9
        )

        value = await agent.evaluate_proof(valid_proof)
        assert value > 0.5

        # Failed proof
        failed_proof = LeanProof(
            theorem_id="test",
            tactics=[],
            lean_code="",
            status=ProofStatus.FAILED,
            confidence=0.0
        )

        value = await agent.evaluate_proof(failed_proof)
        assert value == 0.0

    def test_update_performance(self, agent):
        """Test performance tracking"""
        agent.update_performance({
            "theorem_id": "test",
            "strategy_used": "direct_proof",
            "success": True,
            "proof_length": 3,
            "reward": 1.0
        })

        assert len(agent.performance_history) == 1

        # Strategy success rate should update
        strategy = next(
            s for s in agent.known_strategies
            if s.name == "direct_proof"
        )
        assert strategy.success_rate > 0


class TestLeanSelfPlayGame:
    """Tests for self-play game"""

    @pytest.fixture
    def game(self, sample_theorem):
        """Create a game for testing"""
        verifier = Lean4Verifier()
        agent = LeanProofAgent(
            agent_id="test",
            llm_config={},
            verifier=verifier
        )
        return LeanSelfPlayGame(sample_theorem, agent, verifier)

    @pytest.mark.asyncio
    async def test_game_creation(self, game, sample_theorem):
        """Test game creation"""
        assert game.theorem.id == sample_theorem.id
        assert game.reward == 0.0
        assert game.value_estimate == 0.0

    @pytest.mark.asyncio
    async def test_play_game(self, game):
        """Test playing a game"""
        experience = await game.play()

        assert isinstance(experience, LeanProofExperience)
        assert experience.theorem.id == game.theorem.id
        assert experience.reward >= 0.0

    def test_calculate_reward(self, game, sample_theorem):
        """Test reward calculation"""
        # Valid proof
        game.proof = LeanProof(
            theorem_id=sample_theorem.id,
            tactics=[LeanTactic(name="rfl")],
            lean_code="rfl",
            status=ProofStatus.VERIFIED,
            confidence=0.9,
            generation_time=1.0,
            verification_time=0.5
        )

        reward = game._calculate_reward()
        assert reward > 0.5

        # Failed proof
        game.proof = LeanProof(
            theorem_id=sample_theorem.id,
            tactics=[],
            lean_code="",
            status=ProofStatus.FAILED,
            confidence=0.0
        )

        reward = game._calculate_reward()
        assert reward >= 0.0


class TestLeanSelfPlayEngine:
    """Tests for self-play engine"""

    @pytest.fixture
    def engine(self):
        """Create an engine for testing"""
        return LeanSelfPlayEngine(
            leanaide_url="http://localhost:7654",
            buffer_capacity=100,
            max_concurrent_games=2
        )

    def test_engine_creation(self, engine):
        """Test engine creation"""
        assert engine.leanaide_url == "http://localhost:7654"
        assert engine.buffer.capacity == 100
        assert engine.iteration_count == 0

    @pytest.mark.asyncio
    async def test_run_self_play(self, engine):
        """Test running self-play"""
        theorem = "∀ n : Nat, n + 0 = n"
        games = 3

        proof = await engine.run_self_play(theorem, games)

        assert proof is not None
        assert engine.iteration_count == games
        assert len(engine.buffer.buffer) == games

    @pytest.mark.asyncio
    async def test_run_batch_self_play(self, engine):
        """Test batch self-play"""
        theorems = [
            "∀ n : Nat, n + 0 = n",
            "∀ a b : Nat, a + b = b + a"
        ]

        results = await engine.run_batch_self_play(theorems, games_per_theorem=2)

        assert len(results) == len(theorems)
        assert engine.iteration_count == len(theorems) * 2

    @pytest.mark.asyncio
    async def test_train_from_buffer(self, engine):
        """Test training from buffer"""
        # First, add some experiences by playing games
        await engine.run_self_play("∀ n : Nat, n + 0 = n", games=5)

        # Then train
        metrics = await engine.train_from_buffer(
            batch_size=4,
            iterations=3
        )

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.total_games == 5
        assert len(engine.metrics_history) > 0

    def test_get_training_progress(self, engine):
        """Test getting training progress"""
        progress = engine.get_training_progress()

        assert "iteration" in progress
        assert "status" in progress
        assert progress["iteration"] == 0

    def test_save_load_checkpoint(self, engine, tmp_path):
        """Test checkpoint save/load"""
        # Run some games
        asyncio.run(engine.run_self_play("∀ n : Nat, n + 0 = n", games=3))

        # Save checkpoint
        filepath = tmp_path / "checkpoint.json"
        engine.save_checkpoint(str(filepath))

        # Load into new engine
        new_engine = LeanSelfPlayEngine(
            leanaide_url="http://localhost:7654"
        )
        new_engine.load_checkpoint(str(filepath))

        assert new_engine.iteration_count == engine.iteration_count


# ============================================================================
# Example Usage Tests
# ============================================================================

class TestExampleUsage:
    """Tests demonstrating example usage"""

    @pytest.mark.asyncio
    async def test_basic_self_play_workflow(self):
        """Test basic self-play workflow"""
        # Create engine
        engine = LeanSelfPlayEngine(
            leanaide_url="http://localhost:7654",
            buffer_capacity=50
        )

        try:
            # Define theorem
            theorem = "∀ n : Nat, n + 0 = n"

            # Run self-play
            best_proof = await engine.run_self_play(
                theorem=theorem,
                games=5
            )

            # Check results
            assert best_proof is not None
            assert len(engine.buffer.buffer) == 5

            # Get progress
            progress = engine.get_training_progress()
            assert progress["total_games"] == 5

        finally:
            await engine.close()

    @pytest.mark.asyncio
    async def test_training_loop(self):
        """Test complete training loop"""
        engine = LeanSelfPlayEngine(
            leanaide_url="http://localhost:7654"
        )

        try:
            # Training loop
            for epoch in range(3):
                # Self-play phase
                theorems = [
                    "∀ n : Nat, n + 0 = n",
                    "∀ a b : Nat, a + b = b + a"
                ]

                await engine.run_batch_self_play(
                    theorems=theorems,
                    games_per_theorem=3
                )

                # Training phase
                metrics = await engine.train_from_buffer(
                    batch_size=4,
                    iterations=5
                )

                print(f"Epoch {epoch + 1}: Success rate = {metrics.success_rate:.2%}")

                assert metrics is not None

        finally:
            await engine.close()

    @pytest.mark.asyncio
    async def test_custom_strategy_usage(self):
        """Test using custom proof strategies"""
        engine = LeanSelfPlayEngine()

        try:
            # Add custom strategy
            custom_strategy = LeanProofStrategy(
                name="my_custom_strategy",
                tactic_sequence=["intro", "simp", "rfl"],
                description="My custom proof strategy",
                适用领域=["algebra"],
                success_rate=0.7
            )

            engine.agent.known_strategies.append(custom_strategy)

            # Run self-play (will use custom strategy)
            await engine.run_self_play(
                theorem="∀ n : Nat, n + 0 = n",
                games=2
            )

            # Check that strategy was used
            strategy_names = [
                exp.strategy_used for exp in engine.buffer.buffer
            ]
            assert "my_custom_strategy" in strategy_names

        finally:
            await engine.close()


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and stress tests"""

    @pytest.mark.asyncio
    async def test_concurrent_games(self):
        """Test running multiple concurrent games"""
        engine = LeanSelfPlayEngine(
            max_concurrent_games=5
        )

        try:
            theorems = [
                f"∀ n : Nat, n + {i} = {i} + n"
                for i in range(10)
            ]

            start_time = asyncio.get_event_loop().time()

            results = await engine.run_batch_self_play(
                theorems=theorems,
                games_per_theorem=3
            )

            elapsed = asyncio.get_event_loop().time() - start_time

            print(f"\nProcessed {len(theorems)} theorems in {elapsed:.2f}s")
            print(f"Average time per theorem: {elapsed / len(theorems):.2f}s")

            assert len(results) == len(theorems)

        finally:
            await engine.close()

    @pytest.mark.asyncio
    async def test_large_buffer(self):
        """Test handling large experience buffers"""
        engine = LeanSelfPlayEngine(buffer_capacity=1000)

        try:
            # Generate many experiences
            for i in range(100):
                await engine.run_self_play(
                    theorem=f"∀ n : Nat, n + {i} = {i} + n",
                    games=10
                )

            # Buffer should handle it
            stats = engine.buffer.get_statistics()
            assert stats["size"] == 1000

            # Sampling should still work
            batch = engine.buffer.sample(batch_size=32)
            assert len(batch) == 32

        finally:
            await engine.close()


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
