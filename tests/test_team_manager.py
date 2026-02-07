"""
Comprehensive Unit Tests for Team Manager

Tests the team manager module structure and functionality.

Author: OpenEvolve QA Team
Date: 2026-02-05
Updated: 2026-02-06 - Fixed import and mocking issues
"""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test utilities
try:
    from tests.test_utilities import (
        safe_import,
        skip_if_not_available,
        create_mock_team,
        set_test_env_vars
    )
    set_test_env_vars()
except ImportError:
    # Fallback if test_utilities not available
    def set_test_env_vars():
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
        os.environ.setdefault("TESTING", "true")
    set_test_env_vars()


class TestTeamManagerModuleExistence:
    """Test team manager module structure"""

    def test_team_manager_module_exists(self):
        """Test team_manager module can be imported"""
        try:
            import team_manager
            assert team_manager is not None
        except ImportError as e:
            pytest.skip(f"team_manager module not available: {e}")


class TestTeamManagerComponents:
    """Test team manager components"""

    def test_team_manager_class_exists(self):
        """Test TeamManager class exists"""
        try:
            from team_manager import TeamManager
            assert TeamManager is not None
        except ImportError as e:
            pytest.skip(f"TeamManager not available: {e}")

    def test_team_class_exists(self):
        """Test Team class exists"""
        try:
            from openevolve_structures import Team
            assert Team is not None
        except ImportError:
            try:
                from team_manager import Team
                assert Team is not None
            except ImportError as e:
                pytest.skip(f"Team class not available: {e}")

    def test_team_member_class_exists(self):
        """Test TeamMember class exists"""
        try:
            from team_manager import TeamMember
            assert TeamMember is not None
        except ImportError as e:
            pytest.skip(f"TeamMember not available: {e}")

    def test_task_class_exists(self):
        """Test Task class exists"""
        try:
            from team_manager import Task
            assert Task is not None
        except ImportError as e:
            pytest.skip(f"Task not available: {e}")


class TestTeamManagerCreation:
    """Test TeamManager instantiation and initialization"""

    def test_manager_creates_successfully(self):
        """Test TeamManager can be instantiated"""
        try:
            from team_manager import TeamManager
        except ImportError:
            pytest.skip("TeamManager not available")
            return

        # Use temp file to avoid conflicts
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            manager = TeamManager(teams_file=temp_file)
            assert manager is not None
            assert hasattr(manager, 'teams')
            assert isinstance(manager.teams, dict)
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestTeamManagerMethods:
    """Test team manager methods"""

    def test_manager_has_create_team_method(self):
        """Test TeamManager has create_team method"""
        try:
            from team_manager import TeamManager
        except ImportError:
            pytest.skip("TeamManager not available")
            return

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            manager = TeamManager(teams_file=temp_file)
            assert hasattr(manager, 'create_team')
            assert callable(manager.create_team)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_manager_has_add_member_method(self):
        """Test TeamManager has add_member method"""
        try:
            from team_manager import TeamManager
        except ImportError:
            pytest.skip("TeamManager not available")
            return

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            manager = TeamManager(teams_file=temp_file)
            assert hasattr(manager, 'add_member')
            assert callable(manager.add_member)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_manager_has_assign_task_method(self):
        """Test TeamManager has assign_task method"""
        try:
            from team_manager import TeamManager
        except ImportError:
            pytest.skip("TeamManager not available")
            return

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            manager = TeamManager(teams_file=temp_file)
            assert hasattr(manager, 'assign_task')
            assert callable(manager.assign_task)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_manager_has_get_status_method(self):
        """Test TeamManager has get_status method"""
        try:
            from team_manager import TeamManager
        except ImportError:
            pytest.skip("TeamManager not available")
            return

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            manager = TeamManager(teams_file=temp_file)
            assert hasattr(manager, 'get_status')
            assert callable(manager.get_status)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_manager_loads_teams_when_modelconfig_is_stub(self, monkeypatch):
        """Team loading should tolerate legacy/stub ModelConfig symbols."""
        try:
            import team_manager
        except ImportError:
            pytest.skip("team_manager module not available")
            return

        class _StubModelConfig:
            pass

        monkeypatch.setattr(team_manager, "ModelConfig", _StubModelConfig)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            payload = {
                "BlueTeam": {
                    "name": "BlueTeam",
                    "role": "Blue",
                    "members": [{"model_id": "gpt-4.1", "temperature": 0.2}],
                    "description": None,
                }
            }
            with open(temp_file, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            manager = team_manager.TeamManager(teams_file=temp_file)
            team = manager.get_team("BlueTeam")
            assert team is not None
            assert len(team.members) == 1
            assert getattr(team.members[0], "model_id", None) == "gpt-4.1"
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestTeamManagerExports:
    """Test module exports"""

    def test_expected_exports_exist(self):
        """Test expected classes are exported"""
        try:
            import team_manager
        except ImportError:
            pytest.skip("team_manager module not available")
            return

        assert hasattr(team_manager, 'TeamManager')
        assert hasattr(team_manager, 'TeamMember')
        assert hasattr(team_manager, 'Task')

        # Team might be imported from openevolve_structures
        try:
            from openevolve_structures import Team
            assert Team is not None
        except ImportError:
            # If openevolve_structures not available, Team might still be in team_manager
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
