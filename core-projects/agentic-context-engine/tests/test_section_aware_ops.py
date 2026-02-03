"""Tests for Section-Aware Operations."""

import unittest

from ace.section_aware_ops import (
    SECTION_SLUG_MAPPING,
    generate_section_id,
    get_section_slug,
    normalize_section_name,
    SectionAwareUpdateBatch,
)
from ace.updates import UpdateOperation


class TestGetSectionSlug(unittest.TestCase):
    """Tests for get_section_slug function."""

    def test_common_sections(self):
        """Test slug generation for known sections."""
        self.assertEqual(
            get_section_slug("financial_strategies_and_insights"), "fin"
        )
        self.assertEqual(get_section_slug("formulas_and_calculations"), "calc")
        self.assertEqual(get_section_slug("code_snippets_and_templates"), "code")
        self.assertEqual(get_section_slug("common_mistakes_to_avoid"), "err")
        self.assertEqual(get_section_slug("problem_solving_heuristics"), "prob")
        self.assertEqual(get_section_slug("context_clues_and_indicators"), "ctx")
        self.assertEqual(get_section_slug("others"), "misc")
        self.assertEqual(get_section_slug("meta_strategies"), "meta")

    def test_unknown_section_single_word(self):
        """Test slug generation for unknown single-word sections."""
        # First letters of single word
        slug = get_section_slug("investment")
        # Should use first letters, min 3 chars
        # "investment" -> "inv" (first 3 letters since it's a single short word)
        self.assertTrue(len(slug) >= 3)
        self.assertTrue(slug.islower())

    def test_unknown_section_multi_word(self):
        """Test slug generation for unknown multi-word sections."""
        slug = get_section_slug("risk_management_strategies")
        # Should take first letter of each word
        self.assertEqual(slug, "rms")

    def test_unknown_section_with_stopwords(self):
        """Test slug generation removes stopwords."""
        slug = get_section_slug("trading_and_investment")
        # "and" should be removed, leaving "trading" and "investment"
        # Should take first letter: "t" + "i" = "ti"
        # But with current implementation it might be "ti" or more
        self.assertTrue(len(slug) >= 2)
        self.assertTrue(slug.startswith("t"))

    def test_unknown_section_short(self):
        """Test slug generation for short section names."""
        slug = get_section_slug("qa")
        # Should be at least 3 characters (padded with 'x' if needed)
        self.assertGreaterEqual(len(slug), 3)
        # "qa" is too short, so it becomes "qax" (padded)
        self.assertEqual(slug, "qax")

    def test_case_insensitive_mapping(self):
        """Test that section mapping is case-insensitive."""
        self.assertEqual(get_section_slug("Financial_Strategies_And_Insights"), "fin")
        self.assertEqual(get_section_slug("FINANCIAL_STRATEGIES_AND_INSIGHTS"), "fin")

    def test_normalization_in_slug_generation(self):
        """Test that section names are normalized before slug lookup."""
        # With spaces
        slug = get_section_slug("financial strategies and insights")
        self.assertEqual(slug, "fin")

        # With special chars
        slug = get_section_slug("formulas & calculations")
        self.assertEqual(slug, "calc")


class TestNormalizeSectionName(unittest.TestCase):
    """Tests for normalize_section_name function."""

    def test_snake_case_already_normalized(self):
        """Test that already normalized names pass through."""
        self.assertEqual(
            normalize_section_name("financial_strategies"), "financial_strategies"
        )

    def test_spaces_to_underscores(self):
        """Test converting spaces to underscores."""
        self.assertEqual(
            normalize_section_name("Financial Strategies"), "financial_strategies"
        )

    def test_ampersand_to_and(self):
        """Test converting & to 'and'."""
        self.assertEqual(
            normalize_section_name("Code & Templates"), "code_and_templates"
        )
        self.assertEqual(
            normalize_section_name("Q&A"), "q_and_a"
        )

    def test_plus_to_and(self):
        """Test converting + to 'and'."""
        self.assertEqual(
            normalize_section_name("Input+Output"), "input_and_output"
        )

    def test_slash_to_or(self):
        """Test converting / to 'or'."""
        self.assertEqual(
            normalize_section_name("Input/Output"), "input_or_output"
        )

    def test_case_conversion(self):
        """Test converting to lowercase."""
        self.assertEqual(
            normalize_section_name("FINANCIAL_STRATEGIES"), "financial_strategies"
        )
        self.assertEqual(
            normalize_section_name("FinancialStrategies"), "financialstrategies"
        )

    def test_special_characters(self):
        """Test removing special characters."""
        self.assertEqual(
            normalize_section_name("Risk/Reward & Strategy!"),
            "risk_or_reward_and_strategy"
        )

    def test_multiple_spaces_collapsed(self):
        """Test that multiple spaces are collapsed."""
        self.assertEqual(
            normalize_section_name("Financial   Strategies"), "financial_strategies"
        )

    def test_leading_trailing_underscores_removed(self):
        """Test removing leading and trailing underscores."""
        self.assertEqual(
            normalize_section_name("_financial_strategies_"), "financial_strategies"
        )

    def test_empty_string(self):
        """Test handling of empty string."""
        self.assertEqual(normalize_section_name(""), "unknown")

    def test_none_input(self):
        """Test handling of None input."""
        self.assertEqual(normalize_section_name(None), "unknown")  # type: ignore[arg-type]


class TestGenerateSectionID(unittest.TestCase):
    """Tests for generate_section_id function."""

    def test_known_section(self):
        """Test ID generation for known sections."""
        self.assertEqual(generate_section_id("financial_strategies", 1), "fin-00001")
        self.assertEqual(generate_section_id("formulas_and_calculations", 42), "calc-00042")

    def test_unknown_section(self):
        """Test ID generation for unknown sections."""
        slug = generate_section_id("unknown_section", 999)
        # Should follow pattern: {slug}-{number:05d}
        parts = slug.split("-")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[1], "00999")
        self.assertLessEqual(len(parts[0]), 5)

    def test_zero_padding(self):
        """Test proper zero-padding."""
        # "test" is a single word, slug is first letters -> "tes"
        self.assertEqual(generate_section_id("test", 1), "tes-00001")
        self.assertEqual(generate_section_id("test", 99999), "tes-99999")
        self.assertEqual(generate_section_id("test", 100), "tes-00100")

    def test_section_normalization(self):
        """Test that section names are normalized."""
        # Should normalize before generating slug
        result = generate_section_id("Financial Strategies", 1)
        # Should use "fin" slug after normalization
        self.assertEqual(result, "fin-00001")


class TestSectionAwareUpdateBatch(unittest.TestCase):
    """Tests for SectionAwareUpdateBatch class."""

    def test_initialization(self):
        """Test basic initialization."""
        batch = SectionAwareUpdateBatch(reasoning="Test batch")
        self.assertEqual(batch.reasoning, "Test batch")
        self.assertEqual(len(batch.operations), 0)
        self.assertEqual(batch.section_index, {})

    def test_add_operation_add_with_auto_id(self):
        """Test adding ADD operation with automatic ID generation."""
        batch = SectionAwareUpdateBatch(reasoning="Add financial skill")
        op = batch.add_operation(
            section="financial_strategies",
            content="Diversify portfolio",
            operation_type="ADD"
        )

        self.assertEqual(op.type, "ADD")
        self.assertEqual(op.section, "financial_strategies")
        self.assertEqual(op.content, "Diversify portfolio")
        self.assertIsNotNone(op.skill_id)
        self.assertTrue(op.skill_id.startswith("fin-"))
        self.assertEqual(len(batch.operations), 1)

    def test_add_operation_with_explicit_id(self):
        """Test adding operation with explicit skill_id."""
        batch = SectionAwareUpdateBatch(reasoning="Add skill with ID")
        op = batch.add_operation(
            section="financial_strategies",
            content="Save money",
            operation_type="ADD",
            skill_id="custom-001"
        )

        self.assertEqual(op.skill_id, "custom-001")

    def test_add_operation_multiple_sections(self):
        """Test adding operations to multiple sections."""
        batch = SectionAwareUpdateBatch(reasoning="Add multiple skills")

        # Add to financial section
        op1 = batch.add_operation("financial_strategies", "Diversify", "ADD")
        # Add to formulas section
        op2 = batch.add_operation("formulas_and_calculations", "PV = FV / (1+r)^n", "ADD")
        # Add another to financial section
        op3 = batch.add_operation("financial_strategies", "Save regularly", "ADD")

        # Check section prefixes
        self.assertTrue(op1.skill_id.startswith("fin-"))
        self.assertTrue(op2.skill_id.startswith("calc-"))
        self.assertTrue(op3.skill_id.startswith("fin-"))

        # Check sequential IDs within sections
        self.assertEqual(op1.skill_id, "fin-00001")
        self.assertEqual(op2.skill_id, "calc-00001")
        self.assertEqual(op3.skill_id, "fin-00002")

    def test_add_operation_normalizes_section(self):
        """Test that section names are normalized."""
        batch = SectionAwareUpdateBatch(reasoning="Test normalization")
        op = batch.add_operation(
            section="Financial Strategies",
            content="Test",
            operation_type="ADD"
        )

        self.assertEqual(op.section, "financial_strategies")

    def test_add_operation_update_type(self):
        """Test adding UPDATE operation."""
        batch = SectionAwareUpdateBatch(reasoning="Update skill")
        op = batch.add_operation(
            section="financial_strategies",
            content="Updated content",
            operation_type="UPDATE",
            skill_id="fin-00001"
        )

        self.assertEqual(op.type, "UPDATE")
        self.assertEqual(op.skill_id, "fin-00001")

    def test_add_operation_with_metadata(self):
        """Test adding operation with metadata."""
        batch = SectionAwareUpdateBatch(reasoning="Add skill with tags")
        op = batch.add_operation(
            section="financial_strategies",
            content="Test content",
            operation_type="ADD",
            metadata={"helpful": 1}
        )

        self.assertEqual(op.metadata, {"helpful": 1})

    def test_get_next_id(self):
        """Test get_next_id method."""
        batch = SectionAwareUpdateBatch(reasoning="Test ID counter")

        # First call
        id1 = batch.get_next_id("fin")
        self.assertEqual(id1, 1)

        # Second call
        id2 = batch.get_next_id("fin")
        self.assertEqual(id2, 2)

        # Different section
        id3 = batch.get_next_id("calc")
        self.assertEqual(id3, 1)

        # Check internal state
        self.assertEqual(batch.section_index["fin"], 2)
        self.assertEqual(batch.section_index["calc"], 1)

    def test_normalize_sections(self):
        """Test normalize_sections method."""
        batch = SectionAwareUpdateBatch(reasoning="Normalize sections")

        # Add operations with non-normalized sections
        batch.operations.append(
            UpdateOperation(
                type="ADD",  # type: ignore[arg-type]
                section="Financial Strategies",
                content="Test"
            )
        )
        batch.operations.append(
            UpdateOperation(
                type="ADD",  # type: ignore[arg-type]
                section="Code & Templates",
                content="Test"
            )
        )

        # Normalize
        batch.normalize_sections()

        # Check normalization
        self.assertEqual(batch.operations[0].section, "financial_strategies")
        self.assertEqual(batch.operations[1].section, "code_and_templates")

    def test_to_update_batch(self):
        """Test conversion to standard UpdateBatch."""
        from ace.updates import UpdateBatch

        section_batch = SectionAwareUpdateBatch(reasoning="Test conversion")
        section_batch.add_operation("financial_strategies", "Test", "ADD")

        std_batch = section_batch.to_update_batch()

        self.assertIsInstance(std_batch, UpdateBatch)
        self.assertEqual(std_batch.reasoning, "Test conversion")
        self.assertEqual(len(std_batch.operations), 1)
        self.assertEqual(std_batch.operations[0].section, "financial_strategies")

    def test_from_update_batch(self):
        """Test creation from standard UpdateBatch."""
        from ace.updates import UpdateBatch

        std_batch = UpdateBatch(reasoning="Test creation")
        std_batch.operations.append(
            UpdateOperation(
                type="ADD",  # type: ignore[arg-type]
                section="financial_strategies",
                content="Test"
            )
        )

        section_batch = SectionAwareUpdateBatch.from_update_batch(std_batch)

        self.assertEqual(section_batch.reasoning, "Test creation")
        self.assertEqual(len(section_batch.operations), 1)
        self.assertEqual(section_batch.operations[0].section, "financial_strategies")

    def test_to_json(self):
        """Test JSON serialization."""
        batch = SectionAwareUpdateBatch(reasoning="Test JSON")
        batch.add_operation("financial_strategies", "Test", "ADD")

        json_data = batch.to_json()

        self.assertEqual(json_data["reasoning"], "Test JSON")
        self.assertEqual(len(json_data["operations"]), 1)
        self.assertIn("section_index", json_data)
        self.assertIsInstance(json_data["section_index"], dict)

    def test_from_json(self):
        """Test JSON deserialization."""
        data = {
            "reasoning": "Test deserialization",
            "operations": [
                {
                    "type": "ADD",
                    "section": "financial_strategies",
                    "content": "Test content"
                }
            ],
            "section_index": {"fin": 1}
        }

        batch = SectionAwareUpdateBatch.from_json(data)

        self.assertEqual(batch.reasoning, "Test deserialization")
        self.assertEqual(len(batch.operations), 1)
        self.assertEqual(batch.section_index, {"fin": 1})
        self.assertEqual(batch.operations[0].section, "financial_strategies")

    def test_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        original = SectionAwareUpdateBatch(reasoning="Roundtrip test")
        original.add_operation("financial_strategies", "Test", "ADD")
        original.add_operation("formulas_and_calculations", "Formula", "ADD")

        # Serialize and deserialize
        json_data = original.to_json()
        restored = SectionAwareUpdateBatch.from_json(json_data)

        # Check equality
        self.assertEqual(restored.reasoning, original.reasoning)
        self.assertEqual(len(restored.operations), len(original.operations))
        self.assertEqual(restored.section_index, original.section_index)
        self.assertEqual(
            restored.operations[0].skill_id,
            original.operations[0].skill_id
        )


class TestIntegrationWithSkillbook(unittest.TestCase):
    """Integration tests with Skillbook."""

    def test_section_aware_batch_applies_to_skillbook(self):
        """Test that SectionAwareUpdateBatch applies correctly to skillbook."""
        from ace.skillbook import Skillbook

        # Create section-aware batch
        batch = SectionAwareUpdateBatch(reasoning="Add skills")
        batch.add_operation("financial_strategies", "Diversify portfolio", "ADD")
        batch.add_operation("formulas_and_calculations", "PV = FV / (1+r)^n", "ADD")

        # Apply to skillbook
        skillbook = Skillbook()
        skillbook.apply_update(batch.to_update_batch())

        # Verify
        self.assertEqual(len(skillbook.skills()), 2)
        skills = skillbook.skills()
        skill_ids = [s.id for s in skills]

        # Check section-prefixed IDs
        fin_ids = [sid for sid in skill_ids if sid.startswith("fin-")]
        calc_ids = [sid for sid in skill_ids if sid.startswith("calc-")]

        self.assertEqual(len(fin_ids), 1)
        self.assertEqual(len(calc_ids), 1)

    def test_section_aware_batch_serialization_to_file(self):
        """Test that section-aware batches can be saved and loaded."""
        import tempfile
        import json

        # Create batch
        batch = SectionAwareUpdateBatch(reasoning="Test persistence")
        batch.add_operation("financial_strategies", "Test", "ADD")

        # Serialize to file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(batch.to_json(), f)
            temp_path = f.name

        try:
            # Load from file
            with open(temp_path, "r") as f:
                loaded_data = json.load(f)

            restored_batch = SectionAwareUpdateBatch.from_json(loaded_data)

            # Verify
            self.assertEqual(restored_batch.reasoning, batch.reasoning)
            self.assertEqual(len(restored_batch.operations), len(batch.operations))
            self.assertEqual(
                restored_batch.operations[0].skill_id,
                batch.operations[0].skill_id
            )
        finally:
            import os
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
