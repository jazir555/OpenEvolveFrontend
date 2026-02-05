"""
Validation Commands

Provides commands for validating configurations, profiles, and presets.

Commands:
    all         - Validate current configuration
    config      - Validate specific config file
    profile     - Validate a profile
    preset      - Validate a preset
    env         - Validate environment variables
    check-all   - Run all validation checks
"""

import click
import sys
from pathlib import Path


@click.group()
def validate():
    """
    Validation commands

    Validate configurations, profiles, presets, and environment.

    \b
    Examples:
      evolve validate all
      evolve validate config evolve.config.yaml
      evolve validate profile prod
      evolve validate env
    """
    pass


@validate.command()
@click.option('--config', type=click.Path(exists=True), help='Config file to validate')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--strict', is_flag=True, help='Treat warnings as errors')
def all(config, verbose, strict):
    """
    Validate current configuration

    Validate the configuration file and report any errors.

    \b
    Examples:
      evolve validate all
      evolve validate all --config myconfig.yaml --verbose
      evolve validate all --strict
    """
    from ...unified.config import ConfigManager, ConfigValidator

    try:
        manager = ConfigManager()

        if config:
            cfg = manager.load_config(config_file=config)
            config_file = config
        else:
            try:
                cfg = manager.load_config()
                config_file = "default config"
            except:
                click.echo("[FAIL] No configuration file found. Use 'evolve config init' first.", err=True)
                return 1

        validator = ConfigValidator()
        result = validator.validate(cfg)

        if result.is_valid:
            click.echo(f"[OK] Configuration is valid ({config_file})")

            if verbose and hasattr(result, 'warnings') and result.warnings:
                click.echo("\nWarnings:")
                for warning in result.warnings:
                    click.echo(f"  ⚠ {warning}")

            return 0
        else:
            click.echo(f"[FAIL] Configuration has {len(result.errors)} error(s):", err=True)
            for error in result.errors:
                click.echo(f"  - {error}", err=True)

            if verbose and hasattr(result, 'warnings') and result.warnings:
                click.echo("\nWarnings:")
                for warning in result.warnings:
                    click.echo(f"  ⚠ {warning}")

            return 1

    except Exception as e:
        click.echo(f"[FAIL] Error validating configuration: {e}", err=True)
        return 1


@validate.command('config')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True)
@click.option('--format', type=click.Choice(['text', 'json', 'yaml']), default='text')
def validate_config_file(config_file, verbose, format):
    """
    Validate specific config file

    Check if a configuration file is valid.

    \b
    Examples:
      evolve validate config evolve.config.yaml
      evolve validate config myconfig.yaml --verbose
      evolve validate config myconfig.yaml --format json
    """
    from ...unified.config import ConfigManager, ConfigValidator

    try:
        manager = ConfigManager()
        cfg = manager.load_config(config_file=config_file)

        validator = ConfigValidator()
        result = validator.validate(cfg)

        output = {
            'file': config_file,
            'valid': result.is_valid,
            'errors': result.errors if not result.is_valid else [],
        }

        if verbose and hasattr(result, 'warnings'):
            output['warnings'] = result.warnings

        if format == 'json':
            import json
            click.echo(json.dumps(output, indent=2))

        elif format == 'yaml':
            import yaml
            click.echo(yaml.dump(output, default_flow_style=False))

        else:  # text
            if result.is_valid:
                click.echo(f"[OK] Configuration is valid ({config_file})")
                if verbose and hasattr(result, 'warnings') and result.warnings:
                    click.echo("\nWarnings:")
                    for warning in result.warnings:
                        click.echo(f"  ⚠ {warning}")
            else:
                click.echo(f"[FAIL] Configuration has {len(result.errors)} error(s):", err=True)
                for error in result.errors:
                    click.echo(f"  - {error}", err=True)

        return 0 if result.is_valid else 1

    except Exception as e:
        click.echo(f"[FAIL] Error validating config: {e}", err=True)
        return 1


@validate.command('profile')
@click.argument('profile_name')
@click.option('--verbose', '-v', is_flag=True)
def check_profile(profile_name, verbose):
    """
    Validate a profile

    Check if a profile is valid.

    \b
    Examples:
      evolve validate profile prod
      evolve validate profile myprofile --verbose
    """
    from ...unified.config import ProfileManager, ConfigValidator

    try:
        profile_mgr = ProfileManager()
        validator = ConfigValidator()

        profile_config = profile_mgr.load_profile(profile_name)
        result = validator.validate(profile_config)

        if result.is_valid:
            click.echo(f"[OK] Profile '{profile_name}' is valid")

            if verbose and hasattr(result, 'warnings') and result.warnings:
                click.echo("\nWarnings:")
                for warning in result.warnings:
                    click.echo(f"  ⚠ {warning}")

            return 0
        else:
            click.echo(f"[FAIL] Profile '{profile_name}' has {len(result.errors)} error(s):", err=True)
            for error in result.errors:
                click.echo(f"  - {error}", err=True)
            return 1

    except Exception as e:
        click.echo(f"[FAIL] Error validating profile: {e}", err=True)
        return 1


@validate.command('preset')
@click.argument('preset_name')
@click.option('--verbose', '-v', is_flag=True)
def check_preset(preset_name, verbose):
    """
    Validate a preset

    Check if a preset is valid.

    \b
    Examples:
      evolve validate preset fast
      evolve validate preset mypreset --verbose
    """
    from ...unified.config import PresetManager, ConfigValidator, UnifiedEvolutionConfig

    try:
        preset_mgr = PresetManager()
        validator = ConfigValidator()

        base_config = UnifiedEvolutionConfig()
        config = preset_mgr.apply_preset(preset_name, base_config)
        result = validator.validate(config)

        if result.is_valid:
            click.echo(f"[OK] Preset '{preset_name}' is valid")

            if verbose and hasattr(result, 'warnings') and result.warnings:
                click.echo("\nWarnings:")
                for warning in result.warnings:
                    click.echo(f"  ⚠ {warning}")

            return 0
        else:
            click.echo(f"[FAIL] Preset '{preset_name}' has {len(result.errors)} error(s):", err=True)
            for error in result.errors:
                click.echo(f"  - {error}", err=True)
            return 1

    except Exception as e:
        click.echo(f"[FAIL] Error validating preset: {e}", err=True)
        return 1


@validate.command('env')
@click.option('--verbose', '-v', is_flag=True)
def check_env(verbose):
    """
    Validate environment variables

    Check if all required environment variables are set and valid.

    \b
    Examples:
      evolve validate env
      evolve validate env --verbose
    """
    from ...unified.config import EnvConfigParser

    try:
        parser = EnvConfigParser()
        errors = parser.validate_env_vars()

        if not errors:
            click.echo("[OK] All environment variables are valid")
            return 0
        else:
            click.echo(f"[FAIL] Found {len(errors)} environment variable error(s):", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            return 1

    except Exception as e:
        click.echo(f"[FAIL] Error validating environment: {e}", err=True)
        return 1


@validate.command('check-all')
@click.option('--config', type=click.Path(), help='Config file to validate')
@click.option('--verbose', '-v', is_flag=True)
def check_all(config, verbose):
    """
    Run all validation checks

    Validate configuration, profiles, presets, and environment.

    \b
    Examples:
      evolve validate check-all
      evolve validate check-all --config myconfig.yaml --verbose
    """
    from ...unified.config import ConfigManager, ConfigValidator, ProfileManager, PresetManager, EnvConfigParser

    exit_code = 0
    results = {}

    # Validate config
    try:
        manager = ConfigManager()

        if config:
            cfg = manager.load_config(config_file=config)
        else:
            cfg = manager.load_config()

        validator = ConfigValidator()
        result = validator.validate(cfg)
        results['config'] = 'valid' if result.is_valid else 'invalid'

        if result.is_valid:
            click.echo("[OK] Configuration: valid")
        else:
            click.echo(f"[FAIL] Configuration: invalid ({len(result.errors)} errors)", err=True)
            if verbose:
                for error in result.errors:
                    click.echo(f"  - {error}", err=True)
            exit_code = 1

    except Exception as e:
        click.echo(f"[FAIL] Configuration: error - {e}", err=True)
        results['config'] = 'error'
        exit_code = 1

    # Validate environment
    try:
        parser = EnvConfigParser()
        errors = parser.validate_env_vars()
        results['env'] = 'valid' if not errors else 'invalid'

        if not errors:
            click.echo("[OK] Environment: valid")
        else:
            click.echo(f"[FAIL] Environment: invalid ({len(errors)} errors)", err=True)
            if verbose:
                for error in errors:
                    click.echo(f"  - {error}", err=True)
            exit_code = 1

    except Exception as e:
        click.echo(f"[FAIL] Environment: error - {e}", err=True)
        results['env'] = 'error'
        exit_code = 1

    # Summary
    click.echo()
    total = len(results)
    valid = sum(1 for v in results.values() if v == 'valid')
    click.echo(f"Summary: {valid}/{total} validations passed")

    return exit_code


@validate.command('quick')
@click.argument('config_file', type=click.Path(exists=True))
def quick_check(config_file):
    """
    Quick validation check

    Perform a quick validation check on a configuration file.

    \b
    Examples:
      evolve validate quick evolve.config.yaml
    """
    from ...unified.config import ConfigManager

    try:
        manager = ConfigManager()
        cfg = manager.load_config(config_file=config_file)

        # Quick checks
        errors = []

        # Check required fields
        if not cfg.mode:
            errors.append("Missing required field: mode")

        if not cfg.domain:
            errors.append("Missing required field: domain")

        if cfg.max_iterations <= 0:
            errors.append(f"Invalid max_iterations: {cfg.max_iterations} (must be > 0)")

        if cfg.population_size <= 0:
            errors.append(f"Invalid population_size: {cfg.population_size} (must be > 0)")

        # Check LLM config if models are specified
        if cfg.llm and cfg.llm.models and len(cfg.llm.models) > 0:
            for i, model in enumerate(cfg.llm.models):
                if not model.name:
                    errors.append(f"LLM model {i}: missing model name")

        if errors:
            click.echo(f"[FAIL] Quick check failed ({len(errors)} error(s)):", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            return 1
        else:
            click.echo("[OK] Quick check passed")
            return 0

    except Exception as e:
        click.echo(f"[FAIL] Quick check error: {e}", err=True)
        return 1


@validate.command('schema')
@click.option('--format', type=click.Choice(['json', 'yaml']), default='json')
def show_schema(format):
    """
    Show configuration schema

    Display the JSON schema for the configuration.

    \b
    Examples:
      evolve validate schema
      evolve validate schema --format yaml
    """
    from ...unified.config import UnifiedEvolutionConfig

    try:
        schema = UnifiedEvolutionConfig.schema()

        if format == 'json':
            import json
            click.echo(json.dumps(schema, indent=2))

        elif format == 'yaml':
            import yaml
            click.echo(yaml.dump(schema, default_flow_style=False))

    except Exception as e:
        click.echo(f"[FAIL] Error showing schema: {e}", err=True)
        return 1
