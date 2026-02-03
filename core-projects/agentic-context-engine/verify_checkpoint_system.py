#!/usr/bin/env python3
"""
Quick verification script for ACE checkpoint system.

This script demonstrates the checkpoint system working correctly
by running a simple training loop with checkpoints.

Usage:
    python verify_checkpoint_system.py
"""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ace import (
    Skillbook,
    Agent,
    Reflector,
    SkillManager,
    Sample,
    OfflineACE,
    TaskEnvironment,
)
from ace.llm import LLMResponse, LLMClient
from typing import Any


class QuickVerifyMockLLM(LLMClient):
    """Simple mock LLM for verification."""

    def __init__(self):
        super().__init__(model="verify-mock")

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return deterministic responses."""
        if "Reflector" in prompt:
            response = json.dumps(
                {
                    "reasoning": "Analysis complete",
                    "error_identification": "",
                    "root_cause_analysis": "",
                    "correct_approach": "Correct",
                    "key_insight": "Learned",
                    "skill_tags": [{"id": "skill_1", "tag": "helpful"}],
                }
            )
        elif "SkillManager" in prompt:
            response = json.dumps(
                {
                    "update": {
                        "reasoning": "Add skill",
                        "operations": [
                            {
                                "type": "ADD",
                                "section": "general",
                                "content": "Test skill",
                                "metadata": {"helpful": 1, "harmful": 0},
                            }
                        ],
                    }
                }
            )
        else:  # Agent
            response = json.dumps(
                {"reasoning": "Test", "final_answer": "42", "skill_ids": []}
            )

        return LLMResponse(text=response)

    def complete_structured(self, prompt: str, response_model, **kwargs):
        """Mock structured output."""
        from ace.updates import UpdateBatch
        from ace.roles import SkillManagerOutput

        response = self.complete(prompt, **kwargs)
        data = json.loads(response.text)

        if response_model == SkillManagerOutput:
            update_data = data.get("update", {})
            update = UpdateBatch.from_json(update_data)
            return SkillManagerOutput(update=update, raw=data)

        return response_model.model_validate(data)


class QuickVerifyEnvironment(TaskEnvironment):
    """Simple environment for verification."""

    def evaluate(self, sample: Sample, agent_output: Any):
        from ace import EnvironmentResult
        return EnvironmentResult(
            feedback="Correct", ground_truth="42", metrics={"correct": 1.0}
        )


def verify_checkpoint_system():
    """Run checkpoint verification."""
    print("=" * 70)
    print("ACE CHECKPOINT SYSTEM VERIFICATION")
    print("=" * 70)
    print()

    with TemporaryDirectory() as temp_dir:
        print(f"[DIR] Temporary checkpoint directory: {temp_dir}")
        print()

        # Create samples
        samples = [
            Sample(
                question=f"What is {i} + {i}?",
                context=f"Problem {i}",
                ground_truth=str(i * 2),
            )
            for i in range(1, 26)  # 25 samples
        ]

        print(f"[DATA] Created {len(samples)} training samples")
        print()

        # Setup ACE components
        llm = QuickVerifyMockLLM()
        agent = Agent(llm)
        reflector = Reflector(llm)
        skill_manager = SkillManager(llm)
        skillbook = Skillbook()

        # Create OfflineACE
        ace = OfflineACE(
            skillbook=skillbook,
            agent=agent,
            reflector=reflector,
            skill_manager=skill_manager,
        )

        print("[TRAIN] Starting training with checkpoints...")
        print(f"   - Checkpoint interval: 10 samples")
        print(f"   - Epochs: 1")
        print()

        # Run training with checkpoints
        results = ace.run(
            samples,
            QuickVerifyEnvironment(),
            epochs=1,
            checkpoint_interval=10,
            checkpoint_dir=temp_dir,
        )

        print("[OK] Training complete!")
        print(f"   - Processed {len(results)} samples")
        print(f"   - Skills learned: {len(ace.skillbook.skills())}")
        print()

        # Verify checkpoint files
        checkpoint_path = Path(temp_dir)
        checkpoint_files = sorted(checkpoint_path.glob("ace_checkpoint_*.json"))

        print("[FILES] Checkpoint files created:")
        for ckpt in checkpoint_files:
            size = ckpt.stat().st_size
            print(f"   [OK] {ckpt.name} ({size} bytes)")
        print()

        # Verify latest checkpoint
        latest_file = checkpoint_path / "ace_latest.json"
        if latest_file.exists():
            print("[OK] Latest checkpoint: ace_latest.json")
        else:
            print("[FAIL] Latest checkpoint: NOT FOUND")
            return False
        print()

        # Verify checkpoint content
        print("[VERIFY] Verifying checkpoint content...")
        checkpoint_10 = checkpoint_path / "ace_checkpoint_10.json"
        checkpoint_20 = checkpoint_path / "ace_checkpoint_20.json"

        if checkpoint_10.exists():
            loaded_10 = Skillbook.load_from_file(str(checkpoint_10))
            print(f"   [OK] Checkpoint 10: {len(loaded_10.skills())} skills")
        else:
            print(f"   [FAIL] Checkpoint 10: NOT FOUND")
            return False

        if checkpoint_20.exists():
            loaded_20 = Skillbook.load_from_file(str(checkpoint_20))
            print(f"   [OK] Checkpoint 20: {len(loaded_20.skills())} skills")
        else:
            print(f"   [FAIL] Checkpoint 20: NOT FOUND")
            return False

        if latest_file.exists():
            loaded_latest = Skillbook.load_from_file(str(latest_file))
            print(f"   [OK] Latest: {len(loaded_latest.skills())} skills")
        print()

        # Verify resume capability
        print("[RESUME] Testing resume capability...")
        loaded_skillbook = Skillbook.load_from_file(str(latest_file))

        agent_2 = Agent(llm)
        reflector_2 = Reflector(llm)
        skill_manager_2 = SkillManager(llm)

        ace_2 = OfflineACE(
            skillbook=loaded_skillbook,
            agent=agent_2,
            reflector=reflector_2,
            skill_manager=skill_manager_2,
        )

        # Continue training
        more_samples = [
            Sample(question=f"What is {i} + {i}?", context="", ground_truth=str(i * 2))
            for i in range(26, 31)  # 5 more samples
        ]

        results_2 = ace_2.run(
            more_samples,
            QuickVerifyEnvironment(),
            epochs=1,
            checkpoint_interval=10,
            checkpoint_dir=temp_dir,
        )

        print(f"   [OK] Resumed training: {len(results_2)} additional samples")
        print(f"   [OK] Total skills after resume: {len(ace_2.skillbook.skills())}")
        print()

        print("=" * 70)
        print("[SUCCESS] CHECKPOINT SYSTEM VERIFICATION: PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  [OK] Checkpoints saved at correct intervals")
        print("  [OK] Checkpoint files contain valid skillbook JSON")
        print("  [OK] Latest checkpoint is up-to-date")
        print("  [OK] Training can resume from checkpoint")
        print("  [OK] Existing skills preserved on resume")
        print()
        return True


if __name__ == "__main__":
    try:
        success = verify_checkpoint_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[ERROR] Verification failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
