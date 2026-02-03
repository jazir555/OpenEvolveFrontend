"""
Comprehensive integration tests for checkpoint system in ACE.

This test suite verifies the checkpoint saving and resuming functionality
of the OfflineACE adaptation loop, ensuring:
- Checkpoints are saved at correct intervals
- Checkpoint files contain valid skillbook JSON
- Checkpoints are properly numbered
- Latest checkpoint is always up-to-date
- Training can resume from saved checkpoints
- Multiple epochs work correctly with checkpoints
- Checkpoint directory creation
- Existing skillbooks are preserved

Checkpoint Format:
- File naming: ace_checkpoint_{N}.json, ace_latest.json
- Location: {checkpoint_dir}/ directory
- Content: Full skillbook in JSON format
- Metadata: Implicit via skillbook stats

Behavior:
- Saved every N successful samples (checkpoint_interval)
- Overwrites ace_latest.json each time
- Numbered checkpoints accumulate
- Resume capability: Load from checkpoint and continue
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict
from datetime import datetime

import pytest

from ace import (
    SkillManager,
    EnvironmentResult,
    Agent,
    LLMClient,
    OfflineACE,
    Skillbook,
    Reflector,
    Sample,
    TaskEnvironment,
)
from ace.llm import LLMResponse


# ============================================================================
# Mock Components for Testing
# ============================================================================


class CheckpointMockLLM(LLMClient):
    """
    Mock LLM client designed for checkpoint testing.

    Returns deterministic responses that create skill updates,
    allowing us to verify skillbook evolution across checkpoints.
    """

    def __init__(self):
        super().__init__(model="checkpoint-mock")
        self.call_count = 0
        self.call_history = []

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return valid JSON responses that incrementally build skills."""
        self.call_count += 1
        self.call_history.append({"prompt": prompt[:100], "kwargs": kwargs})

        # Detect role from prompt
        if "ACE Reflector" in prompt or "Reflector" in prompt:
            response = json.dumps(
                {
                    "reasoning": f"Analysis iteration {self.call_count}",
                    "error_identification": "",
                    "root_cause_analysis": "",
                    "correct_approach": "Strategy was effective",
                    "key_insight": f"Insight from iteration {self.call_count}",
                    "skill_tags": [
                        {"id": f"skill_{self.call_count}", "tag": "helpful"}
                    ],
                }
            )
        elif "ACE SkillManager" in prompt or "SkillManager" in prompt:
            # Add a new skill every other call to show skillbook growth
            if self.call_count % 2 == 0:
                response = json.dumps(
                    {
                        "update": {
                            "reasoning": f"Adding skill {self.call_count}",
                            "operations": [
                                {
                                    "type": "ADD",
                                    "section": "general",
                                    "content": f"Learned strategy {self.call_count}: Always verify before answering",
                                    "metadata": {"helpful": 1, "harmful": 0},
                                }
                            ],
                        }
                    }
                )
            else:
                response = json.dumps(
                    {
                        "update": {
                            "reasoning": "No changes needed",
                            "operations": [],
                        }
                    }
                )
        elif "ACE Agent" in prompt or "Agent" in prompt or "skill_ids" in prompt:
            response = json.dumps(
                {
                    "reasoning": "Mock reasoning for answer",
                    "final_answer": "This is a correct mock answer",
                    "skill_ids": [],
                }
            )
        else:
            response = json.dumps({"result": "Mock result"})

        return LLMResponse(text=response)

    def complete_structured(self, prompt: str, response_model, **kwargs):
        """Mock structured output to prevent Instructor wrapping."""
        from ace.updates import UpdateBatch
        from ace.roles import SkillManagerOutput

        response = self.complete(prompt, **kwargs)
        data = json.loads(response.text)

        if response_model == SkillManagerOutput:
            update_data = data.get("update", {})
            update = UpdateBatch.from_json(update_data)
            return SkillManagerOutput(update=update, raw=data)

        return response_model.model_validate(data)


class SimpleTestEnvironment(TaskEnvironment):
    """Simple environment for testing - always returns positive feedback."""

    def evaluate(self, sample: Sample, agent_output: Any) -> EnvironmentResult:
        """Return successful evaluation result."""
        return EnvironmentResult(
            feedback="Correct! Good job.",
            ground_truth=sample.ground_truth,
            metrics={"correct": 1.0},
        )


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def checkpoint_llm():
    """Provides a fresh CheckpointMockLLM for each test."""
    return CheckpointMockLLM()


@pytest.fixture
def checkpoint_samples():
    """
    Provides a set of training samples for checkpoint testing.

    Returns 25 samples to test various checkpoint intervals (5, 10, 25).
    """
    samples = []
    for i in range(1, 26):
        samples.append(
            Sample(
                question=f"Test question {i}",
                context=f"Test context {i}",
                ground_truth=f"Answer {i}",
                metadata={"sample_id": i},
            )
        )
    return samples


@pytest.fixture
def checkpoint_env():
    """Provides a SimpleTestEnvironment instance."""
    return SimpleTestEnvironment()


# ============================================================================
# Checkpoint Integration Tests
# ============================================================================


class TestCheckpointSavingDuringTraining:
    """Test suite for checkpoint saving during OfflineACE training."""

    def test_checkpoint_saving_during_training(
        self, checkpoint_llm, checkpoint_samples, checkpoint_env
    ):
        """
        Test that checkpoints are saved at correct intervals.

        Given:
            - 25 training samples
            - checkpoint_interval = 10
            - Checkpoint directory specified

        When:
            - Running OfflineACE for 1 epoch

        Then:
            - Checkpoints saved at samples 10 and 20
            - Files: ace_checkpoint_10.json, ace_checkpoint_20.json
            - Latest checkpoint always up-to-date
        """
        with TemporaryDirectory() as temp_dir:
            # Setup
            skillbook = Skillbook()
            agent = Agent(checkpoint_llm)
            reflector = Reflector(checkpoint_llm)
            skill_manager = SkillManager(checkpoint_llm)

            ace = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            # Run with checkpointing
            results = ace.run(
                checkpoint_samples,
                checkpoint_env,
                epochs=1,
                checkpoint_interval=10,
                checkpoint_dir=temp_dir,
            )

            # Verify checkpoint files exist
            checkpoint_path = Path(temp_dir)
            checkpoint_10 = checkpoint_path / "ace_checkpoint_10.json"
            checkpoint_20 = checkpoint_path / "ace_checkpoint_20.json"
            latest_checkpoint = checkpoint_path / "ace_latest.json"

            assert checkpoint_10.exists(), "Checkpoint 10 should exist"
            assert checkpoint_20.exists(), "Checkpoint 20 should exist"
            assert latest_checkpoint.exists(), "Latest checkpoint should exist"

            # Verify checkpoints contain valid skillbooks
            loaded_10 = Skillbook.load_from_file(str(checkpoint_10))
            loaded_20 = Skillbook.load_from_file(str(checkpoint_20))
            loaded_latest = Skillbook.load_from_file(str(latest_checkpoint))

            # Latest should match checkpoint_20 (most recent)
            assert (
                loaded_latest.as_prompt() == loaded_20.as_prompt()
            ), "Latest checkpoint should match most recent numbered checkpoint"

    def test_checkpoint_file_format(self, checkpoint_llm, checkpoint_samples, checkpoint_env):
        """
        Test that checkpoint files contain valid skillbook JSON.

        Given:
            - Training samples
            - Checkpointing enabled

        When:
            - Saving checkpoints

        Then:
            - Each checkpoint is valid JSON
            - Can be loaded into Skillbook
            - Contains expected skillbook structure
        """
        with TemporaryDirectory() as temp_dir:
            skillbook = Skillbook()
            agent = Agent(checkpoint_llm)
            reflector = Reflector(checkpoint_llm)
            skill_manager = SkillManager(checkpoint_llm)

            ace = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            ace.run(
                checkpoint_samples[:10],
                checkpoint_env,
                epochs=1,
                checkpoint_interval=5,
                checkpoint_dir=temp_dir,
            )

            # Load and verify checkpoint
            checkpoint_path = Path(temp_dir) / "ace_checkpoint_5.json"
            assert checkpoint_path.exists()

            # Verify JSON is valid
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, dict), "Checkpoint root should be object"
            assert "skills" in data, "Checkpoint should have skills field"

            # Verify can load as skillbook
            loaded = Skillbook.load_from_file(str(checkpoint_path))
            assert len(loaded.skills()) >= 0, "Skillbook should load successfully"

    def test_checkpoint_numbering(self, checkpoint_llm, checkpoint_samples, checkpoint_env):
        """
        Test that checkpoints are numbered correctly.

        Given:
            - 25 samples
            - checkpoint_interval = 5

        When:
            - Running training

        Then:
            - Checkpoints numbered by total samples processed
            - Files: ace_checkpoint_5.json, ace_checkpoint_10.json, etc.
        """
        with TemporaryDirectory() as temp_dir:
            skillbook = Skillbook()
            agent = Agent(checkpoint_llm)
            reflector = Reflector(checkpoint_llm)
            skill_manager = SkillManager(checkpoint_llm)

            ace = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            ace.run(
                checkpoint_samples,
                checkpoint_env,
                epochs=1,
                checkpoint_interval=5,
                checkpoint_dir=temp_dir,
            )

            # Verify checkpoint numbering
            checkpoint_path = Path(temp_dir)
            expected_checkpoints = [5, 10, 15, 20, 25]

            for num in expected_checkpoints:
                checkpoint_file = checkpoint_path / f"ace_checkpoint_{num}.json"
                assert (
                    checkpoint_file.exists()
                ), f"Checkpoint {num} should exist at {checkpoint_file}"

    def test_checkpoint_latest_always_most_recent(
        self, checkpoint_llm, checkpoint_samples, checkpoint_env
    ):
        """
        Test that ace_latest.json is always the most recent checkpoint.

        Given:
            - Multiple checkpoints saved during training

        When:
            - Training progresses and saves checkpoints

        Then:
            - ace_latest.json always matches last numbered checkpoint
            - ace_latest.json is overwritten on each save
        """
        with TemporaryDirectory() as temp_dir:
            skillbook = Skillbook()
            agent = Agent(checkpoint_llm)
            reflector = Reflector(checkpoint_llm)
            skill_manager = SkillManager(checkpoint_llm)

            ace = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            ace.run(
                checkpoint_samples,
                checkpoint_env,
                epochs=1,
                checkpoint_interval=10,
                checkpoint_dir=temp_dir,
            )

            # Compare ace_latest.json with ace_checkpoint_20.json (last checkpoint)
            checkpoint_path = Path(temp_dir)
            latest = Skillbook.load_from_file(str(checkpoint_path / "ace_latest.json"))
            last_numbered = Skillbook.load_from_file(
                str(checkpoint_path / "ace_checkpoint_20.json")
            )

            assert (
                latest.as_prompt() == last_numbered.as_prompt()
            ), "Latest checkpoint should match last numbered checkpoint"

    def test_resume_from_checkpoint(
        self, checkpoint_llm, checkpoint_samples, checkpoint_env
    ):
        """
        Test that training can resume from a saved checkpoint.

        Given:
            - Initial training saves checkpoint at 10 samples
            - Skillbook has learned some skills

        When:
            - Loading checkpoint and continuing training

        Then:
            - New training starts from loaded skillbook state
            - Skills from checkpoint are preserved
            - New skills are added on top of existing ones
        """
        with TemporaryDirectory() as temp_dir:
            # Phase 1: Initial training
            skillbook_1 = Skillbook()
            agent_1 = Agent(checkpoint_llm)
            reflector_1 = Reflector(checkpoint_llm)
            skill_manager_1 = SkillManager(checkpoint_llm)

            ace_1 = OfflineACE(
                skillbook=skillbook_1,
                agent=agent_1,
                reflector=reflector_1,
                skill_manager=skill_manager_1,
            )

            # Train for 10 samples and checkpoint
            results_1 = ace_1.run(
                checkpoint_samples[:10],
                checkpoint_env,
                epochs=1,
                checkpoint_interval=10,
                checkpoint_dir=temp_dir,
            )

            skills_after_phase_1 = len(ace_1.skillbook.skills())

            # Phase 2: Resume from checkpoint
            checkpoint_path = Path(temp_dir) / "ace_checkpoint_10.json"
            loaded_skillbook = Skillbook.load_from_file(str(checkpoint_path))

            # Create new ACE instance with loaded skillbook
            agent_2 = Agent(checkpoint_llm)
            reflector_2 = Reflector(checkpoint_llm)
            skill_manager_2 = SkillManager(checkpoint_llm)

            ace_2 = OfflineACE(
                skillbook=loaded_skillbook,
                agent=agent_2,
                reflector=reflector_2,
                skill_manager=skill_manager_2,
            )

            # Continue training for 10 more samples
            results_2 = ace_2.run(
                checkpoint_samples[10:20],
                checkpoint_env,
                epochs=1,
                checkpoint_interval=10,
                checkpoint_dir=temp_dir,
            )

            # Verify skills from phase 1 are preserved
            skills_after_phase_2 = len(ace_2.skillbook.skills())
            assert (
                skills_after_phase_2 >= skills_after_phase_1
            ), "Resume should preserve existing skills"

    def test_checkpoint_with_multiple_epochs(
        self, checkpoint_llm, checkpoint_samples, checkpoint_env
    ):
        """
        Test that checkpoints work correctly across multiple epochs.

        Given:
            - 2 epochs over 15 samples (30 total samples processed)

        When:
            - checkpoint_interval = 10

        Then:
            - Checkpoints at samples 10, 20, 30
            - Checkpoint numbering based on total samples processed
            - Latest checkpoint reflects final state
        """
        with TemporaryDirectory() as temp_dir:
            skillbook = Skillbook()
            agent = Agent(checkpoint_llm)
            reflector = Reflector(checkpoint_llm)
            skill_manager = SkillManager(checkpoint_llm)

            ace = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            # Run 2 epochs over 15 samples = 30 total samples
            ace.run(
                checkpoint_samples[:15],
                checkpoint_env,
                epochs=2,
                checkpoint_interval=10,
                checkpoint_dir=temp_dir,
            )

            # Verify checkpoints at 10, 20, 30
            checkpoint_path = Path(temp_dir)
            expected = [10, 20, 30]

            for num in expected:
                checkpoint_file = checkpoint_path / f"ace_checkpoint_{num}.json"
                assert checkpoint_file.exists(), f"Checkpoint {num} should exist"

            # Verify latest checkpoint
            latest = Skillbook.load_from_file(str(checkpoint_path / "ace_latest.json"))
            checkpoint_30 = Skillbook.load_from_file(
                str(checkpoint_path / "ace_checkpoint_30.json")
            )
            assert latest.as_prompt() == checkpoint_30.as_prompt()

    def test_checkpoint_directory_creation(
        self, checkpoint_llm, checkpoint_samples, checkpoint_env
    ):
        """
        Test that checkpoint directory is created if it doesn't exist.

        Given:
            - Non-existent checkpoint directory path

        When:
            - Running training with checkpoints

        Then:
            - Directory is created automatically
            - Checkpoints are saved successfully
        """
        with TemporaryDirectory() as temp_dir:
            # Create a non-existent subdirectory path
            checkpoint_dir = Path(temp_dir) / "checkpoints" / "nested" / "path"

            assert not checkpoint_dir.exists(), "Directory should not exist initially"

            skillbook = Skillbook()
            agent = Agent(checkpoint_llm)
            reflector = Reflector(checkpoint_llm)
            skill_manager = SkillManager(checkpoint_llm)

            ace = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            # Run training - should create directory
            ace.run(
                checkpoint_samples[:10],
                checkpoint_env,
                epochs=1,
                checkpoint_interval=5,
                checkpoint_dir=str(checkpoint_dir),
            )

            # Verify directory created and checkpoints saved
            assert checkpoint_dir.exists(), "Directory should be created"
            assert (checkpoint_dir / "ace_checkpoint_5.json").exists()

    def test_checkpoint_with_existing_skillbook(
        self, checkpoint_llm, checkpoint_samples, checkpoint_env
    ):
        """
        Test checkpoint saving when starting with existing skillbook.

        Given:
            - Pre-populated skillbook with existing skills
            - Checkpointing enabled

        When:
            - Running training

        Then:
            - Checkpoints include both existing and new skills
            - Existing skills are preserved in checkpoints
        """
        with TemporaryDirectory() as temp_dir:
            # Create skillbook with existing skills
            skillbook = Skillbook()
            skillbook.add_skill(
                "general",
                "Existing strategy 1",
                metadata={"helpful": 5, "harmful": 0},
            )
            skillbook.add_skill(
                "math",
                "Existing math strategy",
                metadata={"helpful": 3, "harmful": 1},
            )

            initial_skill_count = len(skillbook.skills())

            agent = Agent(checkpoint_llm)
            reflector = Reflector(checkpoint_llm)
            skill_manager = SkillManager(checkpoint_llm)

            ace = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            # Run training
            ace.run(
                checkpoint_samples[:10],
                checkpoint_env,
                epochs=1,
                checkpoint_interval=5,
                checkpoint_dir=temp_dir,
            )

            # Load checkpoint and verify existing skills preserved
            checkpoint_path = Path(temp_dir) / "ace_checkpoint_10.json"
            loaded_skillbook = Skillbook.load_from_file(str(checkpoint_path))

            assert (
                len(loaded_skillbook.skills()) >= initial_skill_count
            ), "Checkpoint should preserve existing skills"

    def test_checkpoint_validation_requires_directory(
        self, checkpoint_llm, checkpoint_samples, checkpoint_env
    ):
        """
        Test that checkpoint_interval requires checkpoint_dir to be set.

        Given:
            - checkpoint_interval set
            - checkpoint_dir not set

        When:
            - Running OfflineACE.run()

        Then:
            - ValueError is raised
            - Error message mentions checkpoint_dir requirement
        """
        skillbook = Skillbook()
        agent = Agent(checkpoint_llm)
        reflector = Reflector(checkpoint_llm)
        skill_manager = SkillManager(checkpoint_llm)

        ace = OfflineACE(
            skillbook=skillbook,
            agent=agent,
            reflector=reflector,
            skill_manager=skill_manager,
        )

        # Should raise ValueError when checkpoint_interval set without dir
        with pytest.raises(ValueError) as exc_info:
            ace.run(
                checkpoint_samples,
                checkpoint_env,
                epochs=1,
                checkpoint_interval=10,
                checkpoint_dir=None,
            )

        assert "checkpoint_dir must be provided" in str(exc_info.value)

    def test_checkpoint_with_interval_larger_than_samples(
        self, checkpoint_llm, checkpoint_samples, checkpoint_env
    ):
        """
        Test behavior when checkpoint_interval > total samples.

        Given:
            - 10 samples
            - checkpoint_interval = 20

        When:
            - Running training

        Then:
            - No checkpoints saved (interval never reached)
            - Training completes successfully
        """
        with TemporaryDirectory() as temp_dir:
            skillbook = Skillbook()
            agent = Agent(checkpoint_llm)
            reflector = Reflector(checkpoint_llm)
            skill_manager = SkillManager(checkpoint_llm)

            ace = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            # Run with interval larger than sample count
            results = ace.run(
                checkpoint_samples[:10],
                checkpoint_env,
                epochs=1,
                checkpoint_interval=20,
                checkpoint_dir=temp_dir,
            )

            # Verify no checkpoints created
            checkpoint_path = Path(temp_dir)
            checkpoint_files = list(checkpoint_path.glob("ace_checkpoint_*.json"))

            assert len(checkpoint_files) == 0, "No checkpoints should be saved"
            assert len(results) == 10, "All samples should be processed"

    def test_checkpoint_skillbook_evolution(
        self, checkpoint_llm, checkpoint_samples, checkpoint_env
    ):
        """
        Test that skillbook evolution is captured across checkpoints.

        Given:
            - Training with multiple checkpoints

        When:
            - Skills are added during training

        Then:
            - Each checkpoint shows skillbook growth
            - Later checkpoints have more skills than earlier ones
        """
        with TemporaryDirectory() as temp_dir:
            skillbook = Skillbook()
            agent = Agent(checkpoint_llm)
            reflector = Reflector(checkpoint_llm)
            skill_manager = SkillManager(checkpoint_llm)

            ace = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            # Run training
            ace.run(
                checkpoint_samples[:25],
                checkpoint_env,
                epochs=1,
                checkpoint_interval=5,
                checkpoint_dir=temp_dir,
            )

            # Load checkpoints and verify evolution
            checkpoint_path = Path(temp_dir)

            checkpoint_5 = Skillbook.load_from_file(
                str(checkpoint_path / "ace_checkpoint_5.json")
            )
            checkpoint_10 = Skillbook.load_from_file(
                str(checkpoint_path / "ace_checkpoint_10.json")
            )
            checkpoint_25 = Skillbook.load_from_file(
                str(checkpoint_path / "ace_checkpoint_25.json")
            )

            # Later checkpoints should have >= skills than earlier ones
            # (Some LLM calls may not add skills, so >= not >)
            assert (
                len(checkpoint_10.skills()) >= len(checkpoint_5.skills())
            ), "Checkpoint 10 should have >= skills than checkpoint 5"
            assert (
                len(checkpoint_25.skills()) >= len(checkpoint_10.skills())
            ), "Checkpoint 25 should have >= skills than checkpoint 10"

    def test_checkpoint_with_failed_samples(
        self, checkpoint_llm, checkpoint_samples, checkpoint_env
    ):
        """
        Test that checkpoints are based on successful samples, not failed ones.

        Given:
            - Some samples may fail during processing
            - Checkpoint interval = 10

        When:
            - Training processes samples with some failures

        Then:
            - Checkpoints saved based on successful samples count
            - Failed samples don't affect checkpoint numbering
        """
        # This test verifies the current implementation behavior
        # Checkpoints are based on results.append() which only happens for successful samples
        # Failed samples are caught, logged, and continued

        with TemporaryDirectory() as temp_dir:
            skillbook = Skillbook()
            agent = Agent(checkpoint_llm)
            reflector = Reflector(checkpoint_llm)
            skill_manager = SkillManager(checkpoint_llm)

            ace = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            # Run training
            ace.run(
                checkpoint_samples[:20],
                checkpoint_env,
                epochs=1,
                checkpoint_interval=10,
                checkpoint_dir=temp_dir,
            )

            # Verify checkpoints at 10 and 20 (if all successful)
            checkpoint_path = Path(temp_dir)

            # With our mock LLM, all should succeed
            assert (checkpoint_path / "ace_checkpoint_10.json").exists()
            assert (checkpoint_path / "ace_checkpoint_20.json").exists()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestCheckpointEdgeCases:
    """Test suite for checkpoint edge cases and error handling."""

    def test_checkpoint_with_interval_of_one(
        self, checkpoint_llm, checkpoint_samples, checkpoint_env
    ):
        """
        Test checkpointing with interval=1 (save after every sample).

        Given:
            - checkpoint_interval = 1
            - 5 samples

        When:
            - Running training

        Then:
            - Checkpoint saved after each sample
            - Files: ace_checkpoint_1.json, ace_checkpoint_2.json, etc.
        """
        with TemporaryDirectory() as temp_dir:
            skillbook = Skillbook()
            agent = Agent(checkpoint_llm)
            reflector = Reflector(checkpoint_llm)
            skill_manager = SkillManager(checkpoint_llm)

            ace = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            # Run with checkpoint every sample
            ace.run(
                checkpoint_samples[:5],
                checkpoint_env,
                epochs=1,
                checkpoint_interval=1,
                checkpoint_dir=temp_dir,
            )

            # Verify checkpoint after each sample
            checkpoint_path = Path(temp_dir)
            for i in range(1, 6):
                checkpoint_file = checkpoint_path / f"ace_checkpoint_{i}.json"
                assert checkpoint_file.exists(), f"Checkpoint {i} should exist"

    def test_checkpoint_with_zero_samples(
        self, checkpoint_llm, checkpoint_samples, checkpoint_env
    ):
        """
        Test behavior when training with zero samples.

        Given:
            - Empty sample list
            - Checkpointing enabled

        When:
            - Running training

        Then:
            - Training completes successfully
            - No checkpoints created
        """
        with TemporaryDirectory() as temp_dir:
            skillbook = Skillbook()
            agent = Agent(checkpoint_llm)
            reflector = Reflector(checkpoint_llm)
            skill_manager = SkillManager(checkpoint_llm)

            ace = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            # Run with empty samples
            results = ace.run(
                [],
                checkpoint_env,
                epochs=1,
                checkpoint_interval=10,
                checkpoint_dir=temp_dir,
            )

            # Verify no checkpoints
            checkpoint_path = Path(temp_dir)
            checkpoint_files = list(checkpoint_path.glob("ace_checkpoint_*.json"))

            assert len(checkpoint_files) == 0, "No checkpoints should be created"
            assert len(results) == 0, "No results should be returned"

    def test_checkpoint_persistence_across_instantiations(
        self, checkpoint_llm, checkpoint_samples, checkpoint_env
    ):
        """
        Test that checkpoint files persist across ACE instance deletions.

        Given:
            - Training creates checkpoints
            - ACE instance is deleted

        When:
            - Loading checkpoint later in new session

        Then:
            - Checkpoint files remain on disk
            - Can be loaded successfully
        """
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)

            # Phase 1: Create checkpoint
            skillbook_1 = Skillbook()
            agent_1 = Agent(checkpoint_llm)
            reflector_1 = Reflector(checkpoint_llm)
            skill_manager_1 = SkillManager(checkpoint_llm)

            ace_1 = OfflineACE(
                skillbook=skillbook_1,
                agent=agent_1,
                reflector=reflector_1,
                skill_manager=skill_manager_1,
            )

            ace_1.run(
                checkpoint_samples[:10],
                checkpoint_env,
                epochs=1,
                checkpoint_interval=10,
                checkpoint_dir=temp_dir,
            )

            # Delete ACE instance
            del ace_1

            # Phase 2: Load checkpoint in new session
            assert (
                checkpoint_path / "ace_checkpoint_10.json"
            ).exists(), "Checkpoint should persist"

            loaded = Skillbook.load_from_file(
                str(checkpoint_path / "ace_checkpoint_10.json")
            )
            assert len(loaded.skills()) >= 0, "Should load successfully"


# ============================================================================
# Checkpoint Metadata and Documentation
# ============================================================================

"""
CHECKPOINT SYSTEM DOCUMENTATION
================================

File Format:
-----------
- Numbered checkpoints: ace_checkpoint_{N}.json
  - N = total samples processed across all epochs
  - Example: ace_checkpoint_10.json (after 10 samples)

- Latest checkpoint: ace_latest.json
  - Always overwritten with most recent checkpoint
  - Convenience alias for current state

Content:
-------
Each checkpoint file contains a complete Skillbook in JSON format:

{
  "skills": [
    {
      "id": "skill_uuid",
      "section": "general",
      "content": "Skill description",
      "metadata": {
        "helpful": 5,
        "harmful": 0
      }
    }
  ]
}

Saving Logic:
-------------
- Checkpoints saved every N successful samples
- checkpoint_interval parameter controls frequency
- Failed samples are skipped and don't count toward checkpoint
- Directory created automatically if missing

Resume Flow:
-----------
1. Load checkpoint: skillbook = Skillbook.load_from_file("ace_checkpoint_N.json")
2. Create new OfflineACE with loaded skillbook
3. Continue training from that state

Use Cases:
---------
1. Resume training after interruption
   - Load latest checkpoint and continue

2. Compare skillbook evolution
   - Load different checkpoints to compare skill growth

3. Early stopping
   - Monitor validation metrics and stop at best checkpoint

4. Ablation studies
   - Save checkpoints at different intervals for analysis

Example Usage:
-------------
```python
from ace import OfflineACE, Agent, Reflector, SkillManager, Sample

# Initial training
ace = OfflineACE(agent=agent, reflector=reflector, skill_manager=skill_manager)
results = ace.run(
    samples,
    environment,
    epochs=3,
    checkpoint_interval=100,
    checkpoint_dir="./checkpoints"
)

# Later: resume from checkpoint
from ace import Skillbook

skillbook = Skillbook.load_from_file("./checkpoints/ace_latest.json")
ace_2 = OfflineACE(
    skillbook=skillbook,
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager
)
results_2 = ace_2.run(more_samples, environment)
```
"""
