"""
Unit tests for OpenEvolve database

Tests the program database functionality including:
- Program storage and retrieval
- Evolutionary tree tracking
- Metadata management
- Query functionality
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
import json


class TestProgramDatabase:
    """Test cases for program database"""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create temporary database path"""
        return tmp_path / "test_programs.db"

    @pytest.fixture
    def database(self, temp_db_path):
        """Create ProgramDatabase instance"""
        from openevolve.database import ProgramDatabase
        return ProgramDatabase(db_path=str(temp_db_path))

    @pytest.fixture
    def sample_program(self):
        """Create sample program data"""
        return {
            "code": "def solve():\n    return 42",
            "fitness": 42.0,
            "generation": 0,
            "parent_id": None,
            "metadata": {
                "llm_model": "gpt-4",
                "timestamp": datetime.now().isoformat()
            }
        }

    def test_database_initialization(self, temp_db_path):
        """Test database can be initialized"""
        from openevolve.database import ProgramDatabase

        db = ProgramDatabase(db_path=str(temp_db_path))

        assert db is not None
        assert db.db_path == str(temp_db_path)
        assert Path(temp_db_path).exists()

    def test_database_creates_file(self, temp_db_path):
        """Test database creates file on initialization"""
        from openevolve.database import ProgramDatabase

        assert not temp_db_path.exists()

        db = ProgramDatabase(db_path=str(temp_db_path))

        assert temp_db_path.exists()

    def test_database_add_program(self, database, sample_program):
        """Test adding program to database"""
        program_id = database.add_program(**sample_program)

        assert program_id is not None
        assert isinstance(program_id, str)

    def test_database_retrieve_program(self, database, sample_program):
        """Test retrieving program from database"""
        program_id = database.add_program(**sample_program)

        retrieved = database.get_program(program_id)

        assert retrieved is not None
        assert retrieved["code"] == sample_program["code"]
        assert retrieved["fitness"] == sample_program["fitness"]
        assert retrieved["generation"] == sample_program["generation"]

    def test_database_retrieve_nonexistent_program(self, database):
        """Test retrieving non-existent program returns None"""
        result = database.get_program("nonexistent_id")

        assert result is None

    def test_database_get_best_program(self, database):
        """Test getting best program from database"""
        # Add multiple programs with different fitness
        database.add_program(code="def solve(): return 100", fitness=100.0, generation=0)
        database.add_program(code="def solve(): return 50", fitness=50.0, generation=0)
        database.add_program(code="def solve(): return 10", fitness=10.0, generation=0)

        best = database.get_best_program()

        assert best is not None
        assert best["fitness"] == 10.0

    def test_database_get_best_by_generation(self, database):
        """Test getting best program for specific generation"""
        # Add programs to different generations
        database.add_program(code="def solve(): return 100", fitness=100.0, generation=0)
        database.add_program(code="def solve(): return 50", fitness=50.0, generation=0)
        database.add_program(code="def solve(): return 75", fitness=75.0, generation=1)
        database.add_program(code="def solve(): return 25", fitness=25.0, generation=1)

        best_gen_0 = database.get_best_program(generation=0)
        best_gen_1 = database.get_best_program(generation=1)

        assert best_gen_0["fitness"] == 50.0
        assert best_gen_1["fitness"] == 25.0

    def test_database_get_programs_by_generation(self, database):
        """Test getting all programs from a generation"""
        database.add_program(code="prog1", fitness=100.0, generation=0)
        database.add_program(code="prog2", fitness=50.0, generation=0)
        database.add_program(code="prog3", fitness=75.0, generation=1)

        gen_0_programs = database.get_programs_by_generation(0)

        assert len(gen_0_programs) == 2
        assert all(p["generation"] == 0 for p in gen_0_programs)

    def test_database_parent_child_relationship(self, database):
        """Test parent-child relationship tracking"""
        parent_id = database.add_program(
            code="def solve(): return 100",
            fitness=100.0,
            generation=0
        )

        child_id = database.add_program(
            code="def solve(): return 50",
            fitness=50.0,
            generation=1,
            parent_id=parent_id
        )

        child = database.get_program(child_id)

        assert child["parent_id"] == parent_id

    def test_database_get_ancestry(self, database):
        """Test getting program ancestry"""
        # Create chain: grandparent -> parent -> child
        grandparent_id = database.add_program(
            code="prog0",
            fitness=100.0,
            generation=0
        )

        parent_id = database.add_program(
            code="prog1",
            fitness=50.0,
            generation=1,
            parent_id=grandparent_id
        )

        child_id = database.add_program(
            code="prog2",
            fitness=25.0,
            generation=2,
            parent_id=parent_id
        )

        ancestry = database.get_ancestry(child_id)

        assert len(ancestry) == 3
        assert ancestry[0]["id"] == grandparent_id
        assert ancestry[1]["id"] == parent_id
        assert ancestry[2]["id"] == child_id

    def test_database_get_children(self, database):
        """Test getting children of a program"""
        parent_id = database.add_program(
            code="parent",
            fitness=100.0,
            generation=0
        )

        database.add_program(code="child1", fitness=50.0, generation=1, parent_id=parent_id)
        database.add_program(code="child2", fitness=25.0, generation=1, parent_id=parent_id)
        database.add_program(code="unrelated", fitness=75.0, generation=1)

        children = database.get_children(parent_id)

        assert len(children) == 2
        assert all(c["parent_id"] == parent_id for c in children)

    def test_database_count_programs(self, database):
        """Test counting programs in database"""
        assert database.get_program_count() == 0

        database.add_program(code="prog1", fitness=100.0, generation=0)
        database.add_program(code="prog2", fitness=50.0, generation=0)

        assert database.get_program_count() == 2

    def test_database_count_by_generation(self, database):
        """Test counting programs by generation"""
        database.add_program(code="prog1", fitness=100.0, generation=0)
        database.add_program(code="prog2", fitness=50.0, generation=0)
        database.add_program(code="prog3", fitness=75.0, generation=1)

        assert database.get_program_count(generation=0) == 2
        assert database.get_program_count(generation=1) == 1

    def test_database_metadata_storage(self, database):
        """Test storing and retrieving metadata"""
        metadata = {
            "llm_model": "gpt-4",
            "temperature": 0.7,
            "timestamp": datetime.now().isoformat(),
            "custom_field": "custom_value"
        }

        program_id = database.add_program(
            code="def solve(): return 42",
            fitness=42.0,
            generation=0,
            metadata=metadata
        )

        program = database.get_program(program_id)

        assert program["metadata"]["llm_model"] == "gpt-4"
        assert program["metadata"]["custom_field"] == "custom_value"


class TestDatabaseQueries:
    """Test database query functionality"""

    @pytest.fixture
    def populated_database(self, tmp_path):
        """Create database populated with test data"""
        from openevolve.database import ProgramDatabase

        db = ProgramDatabase(db_path=str(tmp_path / "test.db"))

        # Add test programs
        programs = [
            {"code": "prog0", "fitness": 100.0, "generation": 0},
            {"code": "prog1", "fitness": 50.0, "generation": 0},
            {"code": "prog2", "fitness": 75.0, "generation": 1},
            {"code": "prog3", "fitness": 25.0, "generation": 1},
            {"code": "prog4", "fitness": 10.0, "generation": 2},
        ]

        for prog in programs:
            db.add_program(**prog)

        return db

    def test_database_get_all_programs(self, populated_database):
        """Test getting all programs"""
        all_programs = populated_database.get_all_programs()

        assert len(all_programs) == 5

    def test_database_get_fitness_range(self, populated_database):
        """Test getting programs within fitness range"""
        programs = populated_database.get_programs_by_fitness_range(20.0, 80.0)

        assert len(programs) == 2
        assert all(20.0 <= p["fitness"] <= 80.0 for p in programs)

    def test_database_get_best_n(self, populated_database):
        """Test getting top N programs"""
        best_3 = populated_database.get_best_programs(n=3)

        assert len(best_3) == 3
        # Should be sorted by fitness
        assert best_3[0]["fitness"] <= best_3[1]["fitness"]
        assert best_3[1]["fitness"] <= best_3[2]["fitness"]
        assert best_3[0]["fitness"] == 10.0

    def test_database_get_generation_stats(self, populated_database):
        """Test getting statistics for a generation"""
        stats = populated_database.get_generation_stats(generation=0)

        assert stats["count"] == 2
        assert stats["best_fitness"] == 50.0
        assert stats["worst_fitness"] == 100.0
        assert stats["avg_fitness"] == 75.0


class TestDatabasePersistence:
    """Test database persistence and loading"""

    def test_database_persists_to_disk(self, tmp_path):
        """Test database writes to disk"""
        from openevolve.database import ProgramDatabase

        db_path = tmp_path / "persist.db"

        # Create database and add program
        db1 = ProgramDatabase(db_path=str(db_path))
        db1.add_program(code="persistent", fitness=42.0, generation=0)

        # Create new database instance pointing to same file
        db2 = ProgramDatabase(db_path=str(db_path))

        # Should retrieve the same program
        all_programs = db2.get_all_programs()
        assert len(all_programs) == 1
        assert all_programs[0]["code"] == "persistent"

    def test_database_survives_restart(self, tmp_path):
        """Test database survives restart"""
        from openevolve.database import ProgramDatabase

        db_path = tmp_path / "restart.db"

        # First session
        db1 = ProgramDatabase(db_path=str(db_path))
        prog_id = db1.add_program(code="restart_test", fitness=100.0, generation=0)

        # "Restart" - new instance
        db2 = ProgramDatabase(db_path=str(db_path))

        # Should still have the program
        retrieved = db2.get_program(prog_id)
        assert retrieved is not None
        assert retrieved["code"] == "restart_test"


class TestDatabaseDeletion:
    """Test database deletion and cleanup"""

    @pytest.fixture
    def database(self, tmp_path):
        """Create database for deletion tests"""
        from openevolve.database import ProgramDatabase
        return ProgramDatabase(db_path=str(tmp_path / "test.db"))

    def test_database_delete_program(self, database):
        """Test deleting a program"""
        prog_id = database.add_program(code="delete_me", fitness=100.0, generation=0)

        assert database.get_program(prog_id) is not None

        database.delete_program(prog_id)

        assert database.get_program(prog_id) is None

    def test_database_clear_generation(self, database):
        """Test clearing all programs from a generation"""
        database.add_program(code="gen0_1", fitness=100.0, generation=0)
        database.add_program(code="gen0_2", fitness=50.0, generation=0)
        database.add_program(code="gen1_1", fitness=75.0, generation=1)

        database.clear_generation(0)

        assert database.get_program_count(generation=0) == 0
        assert database.get_program_count(generation=1) == 1

    def test_database_clear_all(self, database):
        """Test clearing all programs"""
        database.add_program(code="prog1", fitness=100.0, generation=0)
        database.add_program(code="prog2", fitness=50.0, generation=1)

        database.clear_all()

        assert database.get_program_count() == 0


class TestDatabaseEvolutionTracking:
    """Test evolutionary tracking features"""

    @pytest.fixture
    def evolutionary_db(self, tmp_path):
        """Create database for evolutionary tracking tests"""
        from openevolve.database import ProgramDatabase
        return ProgramDatabase(db_path=str(tmp_path / "evo.db"))

    def test_tracks_evolutionary_chain(self, evolutionary_db):
        """Test tracking full evolutionary chain"""
        # Create evolution chain
        ids = []
        for i in range(5):
            parent_id = ids[-1] if ids else None
            prog_id = evolutionary_db.add_program(
                code=f"generation_{i}",
                fitness=100.0 - i * 20,
                generation=i,
                parent_id=parent_id
            )
            ids.append(prog_id)

        # Verify chain
        ancestry = evolutionary_db.get_ancestry(ids[-1])

        assert len(ancestry) == 5
        assert ancestry[0]["generation"] == 0
        assert ancestry[-1]["generation"] == 4

    def test_evolutionary_divergence(self, evolutionary_db):
        """Test tracking divergent evolution"""
        # Common ancestor
        ancestor = evolutionary_db.add_program(
            code="ancestor",
            fitness=100.0,
            generation=0
        )

        # Two lineages from ancestor
        lineage1 = evolutionary_db.add_program(
            code="lineage1",
            fitness=50.0,
            generation=1,
            parent_id=ancestor
        )

        lineage2 = evolutionary_db.add_program(
            code="lineage2",
            fitness=60.0,
            generation=1,
            parent_id=ancestor
        )

        # Verify both have same ancestor
        ancestry1 = evolutionary_db.get_ancestry(lineage1)
        ancestry2 = evolutionary_db.get_ancestry(lineage2)

        assert ancestry1[0]["id"] == ancestry2[0]["id"]
        assert ancestry1[0]["id"] == ancestor
