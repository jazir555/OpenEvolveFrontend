"""
Strategy Profiles for OpenEvolve Gauntlet System

Provides configurable strategy profiles that control how the Gauntlet
system operates, allowing different approaches (conservative, balanced,
aggressive, etc.) for different use cases.

Key Features:
- Predefined strategy profiles
- Custom profile creation
- Profile validation
- Profile switching during execution
- Profile performance tracking
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StrategyProfile:
    """A complete strategy profile configuration"""
    name: str
    display_name: str
    description: str
    category: str  # 'preset', 'custom'

    # Decomposition settings
    max_decomposition_depth: int = 3
    min_atomic_size: int = 1
    decomposition_threshold: float = 0.7

    # Gauntlet settings
    max_gauntlet_rounds: int = 3
    blue_team_iterations: int = 1
    red_team_attacks: int = 2
    gold_team_judges: int = 1

    # Quality settings
    pass_threshold: float = 0.75
    quality_weight: float = 0.5
    speed_weight: float = 0.5

    # Performance settings
    parallel_enabled: bool = True
    max_parallelism: int = 4
    cache_enabled: bool = True

    # Difficulty settings
    difficulty_adjustment: bool = False
    target_difficulty: str = "medium"

    # Monitoring settings
    checkpointing_enabled: bool = True
    checkpoint_frequency: str = "major"  # 'major', 'minor', 'all'
    visualization_enabled: bool = False

    # Advanced settings
    fuzzing_enabled: bool = False
    fuzz_iterations: int = 100
    traceability_enabled: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'category': self.category,
            'max_decomposition_depth': self.max_decomposition_depth,
            'min_atomic_size': self.min_atomic_size,
            'decomposition_threshold': self.decomposition_threshold,
            'max_gauntlet_rounds': self.max_gauntlet_rounds,
            'blue_team_iterations': self.blue_team_iterations,
            'red_team_attacks': self.red_team_attacks,
            'gold_team_judges': self.gold_team_judges,
            'pass_threshold': self.pass_threshold,
            'quality_weight': self.quality_weight,
            'speed_weight': self.speed_weight,
            'parallel_enabled': self.parallel_enabled,
            'max_parallelism': self.max_parallelism,
            'cache_enabled': self.cache_enabled,
            'difficulty_adjustment': self.difficulty_adjustment,
            'target_difficulty': self.target_difficulty,
            'checkpointing_enabled': self.checkpointing_enabled,
            'checkpoint_frequency': self.checkpoint_frequency,
            'visualization_enabled': self.visualization_enabled,
            'fuzzing_enabled': self.fuzzing_enabled,
            'fuzz_iterations': self.fuzz_iterations,
            'traceability_enabled': self.traceability_enabled,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyProfile':
        """Create profile from dictionary"""
        return cls(
            name=data['name'],
            display_name=data['display_name'],
            description=data['description'],
            category=data.get('category', 'custom'),
            max_decomposition_depth=data.get('max_decomposition_depth', 3),
            min_atomic_size=data.get('min_atomic_size', 1),
            decomposition_threshold=data.get('decomposition_threshold', 0.7),
            max_gauntlet_rounds=data.get('max_gauntlet_rounds', 3),
            blue_team_iterations=data.get('blue_team_iterations', 1),
            red_team_attacks=data.get('red_team_attacks', 2),
            gold_team_judges=data.get('gold_team_judges', 1),
            pass_threshold=data.get('pass_threshold', 0.75),
            quality_weight=data.get('quality_weight', 0.5),
            speed_weight=data.get('speed_weight', 0.5),
            parallel_enabled=data.get('parallel_enabled', True),
            max_parallelism=data.get('max_parallelism', 4),
            cache_enabled=data.get('cache_enabled', True),
            difficulty_adjustment=data.get('difficulty_adjustment', False),
            target_difficulty=data.get('target_difficulty', 'medium'),
            checkpointing_enabled=data.get('checkpointing_enabled', True),
            checkpoint_frequency=data.get('checkpoint_frequency', 'major'),
            visualization_enabled=data.get('visualization_enabled', False),
            fuzzing_enabled=data.get('fuzzing_enabled', False),
            fuzz_iterations=data.get('fuzz_iterations', 100),
            traceability_enabled=data.get('traceability_enabled', False),
            metadata=data.get('metadata', {}),
        )


class ProfileValidator:
    """Validates strategy profiles"""

    def validate(self, profile: StrategyProfile) -> tuple[bool, List[str]]:
        """
        Validate a strategy profile.

        Args:
            profile: Profile to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        # Validate decomposition settings
        if profile.max_decomposition_depth < 1:
            errors.append("max_decomposition_depth must be at least 1")
        if profile.max_decomposition_depth > 10:
            errors.append("max_decomposition_depth should not exceed 10")

        if profile.min_atomic_size < 1:
            errors.append("min_atomic_size must be at least 1")

        if not (0 <= profile.decomposition_threshold <= 1):
            errors.append("decomposition_threshold must be between 0 and 1")

        # Validate gauntlet settings
        if profile.max_gauntlet_rounds < 1:
            errors.append("max_gauntlet_rounds must be at least 1")
        if profile.max_gauntlet_rounds > 10:
            errors.append("max_gauntlet_rounds should not exceed 10")

        if profile.blue_team_iterations < 1:
            errors.append("blue_team_iterations must be at least 1")
        if profile.red_team_attacks < 0:
            errors.append("red_team_attacks cannot be negative")
        if profile.gold_team_judges < 1:
            errors.append("gold_team_judges must be at least 1")

        # Validate quality settings
        if not (0 <= profile.pass_threshold <= 1):
            errors.append("pass_threshold must be between 0 and 1")

        if not (0 <= profile.quality_weight <= 1):
            errors.append("quality_weight must be between 0 and 1")
        if not (0 <= profile.speed_weight <= 1):
            errors.append("speed_weight must be between 0 and 1")

        if abs(profile.quality_weight + profile.speed_weight - 1.0) > 0.01:
            errors.append("quality_weight and speed_weight must sum to 1.0")

        # Validate performance settings
        if profile.max_parallelism < 1:
            errors.append("max_parallelism must be at least 1")
        if profile.max_parallelism > 20:
            errors.append("max_parallelism should not exceed 20")

        # Validate difficulty
        valid_difficulties = ['very_easy', 'easy', 'medium', 'hard', 'very_hard', 'expert']
        if profile.target_difficulty not in valid_difficulties:
            errors.append(f"target_difficulty must be one of {valid_difficulties}")

        # Validate checkpointing
        valid_frequencies = ['major', 'minor', 'all']
        if profile.checkpoint_frequency not in valid_frequencies:
            errors.append(f"checkpoint_frequency must be one of {valid_frequencies}")

        # Validate fuzzing
        if profile.fuzz_iterations < 1:
            errors.append("fuzz_iterations must be at least 1")
        if profile.fuzz_iterations > 100000:
            errors.append("fuzz_iterations should not exceed 100000")

        is_valid = len(errors) == 0
        return (is_valid, errors)


class ProfileLoader:
    """Loads and manages strategy profiles"""

    def __init__(self, profiles_dir: str = None):
        self.profiles_dir = profiles_dir or "./profiles"
        self.profiles: Dict[str, StrategyProfile] = {}
        self.validator = ProfileValidator()

        # Load preset profiles
        self._load_preset_profiles()

    def _load_preset_profiles(self):
        """Load preset strategy profiles"""

        # CONSERVATIVE: High quality, slow, thorough
        conservative = StrategyProfile(
            name="conservative",
            display_name="Conservative",
            description="High quality focus with thorough testing and validation",
            category="preset",
            max_decomposition_depth=5,
            max_gauntlet_rounds=5,
            blue_team_iterations=2,
            red_team_attacks=5,
            gold_team_judges=3,
            pass_threshold=0.90,
            quality_weight=0.9,
            speed_weight=0.1,
            parallel_enabled=True,
            max_parallelism=2,
            cache_enabled=True,
            fuzzing_enabled=True,
            fuzz_iterations=500,
            traceability_enabled=True,
        )

        # BALANCED: Moderate quality and speed
        balanced = StrategyProfile(
            name="balanced",
            display_name="Balanced",
            description="Balanced approach with moderate quality and speed",
            category="preset",
            max_decomposition_depth=3,
            max_gauntlet_rounds=3,
            blue_team_iterations=1,
            red_team_attacks=2,
            gold_team_judges=1,
            pass_threshold=0.75,
            quality_weight=0.5,
            speed_weight=0.5,
            parallel_enabled=True,
            max_parallelism=4,
            cache_enabled=True,
            fuzzing_enabled=False,
            traceability_enabled=False,
        )

        # AGGRESSIVE: Fast, lower quality threshold
        aggressive = StrategyProfile(
            name="aggressive",
            display_name="Aggressive",
            description="Fast execution with lower quality threshold",
            category="preset",
            max_decomposition_depth=2,
            max_gauntlet_rounds=2,
            blue_team_iterations=1,
            red_team_attacks=1,
            gold_team_judges=1,
            pass_threshold=0.60,
            quality_weight=0.3,
            speed_weight=0.7,
            parallel_enabled=True,
            max_parallelism=8,
            cache_enabled=True,
            fuzzing_enabled=False,
            traceability_enabled=False,
        )

        # FAST: Maximum speed, minimal validation
        fast = StrategyProfile(
            name="fast",
            display_name="Fast",
            description="Maximum speed with minimal validation",
            category="preset",
            max_decomposition_depth=1,
            max_gauntlet_rounds=1,
            blue_team_iterations=1,
            red_team_attacks=0,
            gold_team_judges=1,
            pass_threshold=0.50,
            quality_weight=0.2,
            speed_weight=0.8,
            parallel_enabled=True,
            max_parallelism=10,
            cache_enabled=True,
            checkpointing_enabled=False,
            fuzzing_enabled=False,
            traceability_enabled=False,
        )

        # THOROUGH: Maximum quality, very slow
        thorough = StrategyProfile(
            name="thorough",
            display_name="Thorough",
            description="Maximum quality with comprehensive testing",
            category="preset",
            max_decomposition_depth=7,
            max_gauntlet_rounds=7,
            blue_team_iterations=3,
            red_team_attacks=10,
            gold_team_judges=5,
            pass_threshold=0.95,
            quality_weight=0.95,
            speed_weight=0.05,
            parallel_enabled=True,
            max_parallelism=2,
            cache_enabled=True,
            fuzzing_enabled=True,
            fuzz_iterations=1000,
            traceability_enabled=True,
            checkpointing_enabled=True,
            checkpoint_frequency="minor",
            visualization_enabled=True,
        )

        # Register all presets
        for profile in [conservative, balanced, aggressive, fast, thorough]:
            self.profiles[profile.name] = profile

        logger.info(f"Loaded {len(self.profiles)} preset profiles")

    def load_profile(self, name: str) -> Optional[StrategyProfile]:
        """Load a profile by name"""
        return self.profiles.get(name)

    def register_profile(self, profile: StrategyProfile) -> bool:
        """
        Register a new profile.

        Args:
            profile: Profile to register

        Returns:
            True if registered successfully
        """
        # Validate profile
        is_valid, errors = self.validator.validate(profile)

        if not is_valid:
            logger.error(f"Invalid profile {profile.name}: {errors}")
            return False

        self.profiles[profile.name] = profile
        logger.info(f"Registered profile: {profile.name}")
        return True

    def list_profiles(self) -> List[str]:
        """List all available profile names"""
        return list(self.profiles.keys())

    def get_profile(self, name: str) -> Optional[StrategyProfile]:
        """Get a profile by name"""
        return self.profiles.get(name)

    def save_profile(self, profile: StrategyProfile, filepath: str = None) -> bool:
        """Save profile to file"""
        if filepath is None:
            filepath = Path(self.profiles_dir) / f"{profile.name}.json"
            filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(filepath, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            return False

    def load_profile_from_file(self, filepath: str) -> Optional[StrategyProfile]:
        """Load profile from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            profile = StrategyProfile.from_dict(data)
            return profile
        except Exception as e:
            logger.error(f"Failed to load profile from {filepath}: {e}")
            return None


class ProfileApplier:
    """Applies strategy profiles to the Gauntlet system"""

    def apply_profile(
        self,
        profile: StrategyProfile,
        problem: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply a strategy profile to a problem.

        Args:
            profile: Strategy profile to apply
            problem: Original problem

        Returns:
            Configured problem with profile settings applied
        """
        configured = problem.copy()

        # Apply decomposition settings
        configured['_profile'] = profile.name
        configured['max_decomposition_depth'] = profile.max_decomposition_depth
        configured['min_atomic_size'] = profile.min_atomic_size
        configured['decomposition_threshold'] = profile.decomposition_threshold

        # Apply gauntlet settings
        configured['max_gauntlet_rounds'] = profile.max_gauntlet_rounds
        configured['blue_team_iterations'] = profile.blue_team_iterations
        configured['red_team_attacks'] = profile.red_team_attacks
        configured['gold_team_judges'] = profile.gold_team_judges
        configured['pass_threshold'] = profile.pass_threshold

        # Apply performance settings
        configured['parallel_enabled'] = profile.parallel_enabled
        configured['max_parallelism'] = profile.max_parallelism
        configured['cache_enabled'] = profile.cache_enabled

        # Apply advanced settings
        configured['fuzzing_enabled'] = profile.fuzzing_enabled
        configured['fuzz_iterations'] = profile.fuzz_iterations
        configured['traceability_enabled'] = profile.traceability_enabled
        configured['checkpointing_enabled'] = profile.checkpointing_enabled
        configured['checkpoint_frequency'] = profile.checkpoint_frequency
        configured['visualization_enabled'] = profile.visualization_enabled

        logger.info(
            f"Applied profile '{profile.name}' to problem {problem.get('id', 'unknown')}"
        )

        return configured


class ProfileManager:
    """
    Main interface for strategy profile management.
    """

    def __init__(self, profiles_dir: str = None):
        self.loader = ProfileLoader(profiles_dir)
        self.applier = ProfileApplier()
        self.active_profile: Optional[str] = None

    def get_available_profiles(self) -> List[str]:
        """Get list of available profile names"""
        return self.loader.list_profiles()

    def get_profile(self, name: str) -> Optional[StrategyProfile]:
        """Get a profile by name"""
        return self.loader.get_profile(name)

    def set_active_profile(self, name: str) -> bool:
        """Set the active profile"""
        profile = self.loader.get_profile(name)
        if not profile:
            logger.error(f"Profile not found: {name}")
            return False

        self.active_profile = name
        logger.info(f"Active profile set to: {name}")
        return True

    def get_active_profile(self) -> Optional[str]:
        """Get the active profile name"""
        return self.active_profile

    def apply_profile(
        self,
        problem: Dict[str, Any],
        profile_name: str = None
    ) -> Dict[str, Any]:
        """
        Apply a profile to a problem.

        Args:
            problem: Problem to configure
            profile_name: Profile name (uses active if not specified)

        Returns:
            Configured problem
        """
        profile_name = profile_name or self.active_profile

        if not profile_name:
            logger.warning("No profile specified, using default 'balanced'")
            profile_name = "balanced"

        profile = self.loader.get_profile(profile_name)
        if not profile:
            logger.error(f"Profile not found: {profile_name}")
            return problem

        return self.applier.apply_profile(profile, problem)

    def create_custom_profile(
        self,
        name: str,
        display_name: str,
        description: str,
        settings: Dict[str, Any]
    ) -> bool:
        """
        Create a custom profile.

        Args:
            name: Profile name (unique identifier)
            display_name: Human-readable name
            description: Profile description
            settings: Profile settings

        Returns:
            True if created successfully
        """
        profile = StrategyProfile(
            name=name,
            display_name=display_name,
            description=description,
            category="custom",
            **settings
        )

        return self.loader.register_profile(profile)

    def validate_profile(self, profile: StrategyProfile) -> tuple[bool, List[str]]:
        """Validate a profile"""
        return self.loader.validator.validate(profile)


# Convenience functions
def create_profile_manager(profiles_dir: str = None) -> ProfileManager:
    """Create a profile manager"""
    return ProfileManager(profiles_dir)


def get_preset_profiles() -> Dict[str, Dict[str, Any]]:
    """Get all preset profiles as dictionaries"""
    loader = ProfileLoader()
    return {
        name: profile.to_dict()
        for name, profile in loader.profiles.items()
        if profile.category == "preset"
    }


# Example usage
def demo_strategy_profiles():
    """Demonstration of strategy profiles"""

    manager = create_profile_manager()

    print("\n" + "=" * 60)
    print("Strategy Profiles Demo")
    print("=" * 60)

    # List available profiles
    profiles = manager.get_available_profiles()
    print(f"\nAvailable profiles: {', '.join(profiles)}")

    # Show profile details
    print("\nProfile Details:")
    for profile_name in profiles:
        profile = manager.get_profile(profile_name)
        print(f"\n  {profile.display_name} ({profile.name})")
        print(f"    Description: {profile.description}")
        print(f"    Quality: {profile.quality_weight:.0%}, Speed: {profile.speed_weight:.0%}")
        print(f"    Max Rounds: {profile.max_gauntlet_rounds}")
        print(f"    Pass Threshold: {profile.pass_threshold:.0%}")

    # Apply profile to problem
    problem = {
        'id': 'test_problem',
        'statement': 'Solve this problem',
        'requirements': ['quality']
    }

    # Apply conservative profile
    configured = manager.apply_profile(problem, 'conservative')
    print(f"\nApplied 'conservative' profile:")
    print(f"  Max Rounds: {configured['max_gauntlet_rounds']}")
    print(f"  Pass Threshold: {configured['pass_threshold']}")
    print(f"  Parallelism: {configured['max_parallelism']}")

    # Apply aggressive profile
    configured = manager.apply_profile(problem, 'aggressive')
    print(f"\nApplied 'aggressive' profile:")
    print(f"  Max Rounds: {configured['max_gauntlet_rounds']}")
    print(f"  Pass Threshold: {configured['pass_threshold']}")
    print(f"  Parallelism: {configured['max_parallelism']}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    demo_strategy_profiles()
