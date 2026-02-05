"""
Configuration Management Commands

Provides commands for initializing, validating, and managing configuration files.

Commands:
    init    - Initialize a new configuration file
    validate - Validate configuration file
    list    - List all configuration parameters
    get     - Get parameter value
    set     - Set parameter value
    diff    - Compare two configuration files
    merge   - Merge multiple configuration files
    export  - Export configuration to different format
    import  - Import configuration from file
"""

import click
import json
import yaml
from pathlib import Path
from typing import Any, Dict
from difflib import unified_diff


@click.group()
def config():
    """
    Configuration management commands

    Manage configuration files for the Unified Evolution Engine.

    \b
    Examples:
      evolve config init --format yaml
      evolve config validate evolve.config.yaml
      evolve config get max_iterations
      evolve config set max_iterations 200
    """
    pass


@config.command('init')
@click.option('--format', type=click.Choice(['yaml', 'json', 'toml']), default='yaml',
              help='Output format')
@click.option('--output', type=click.Path(), default='evolve.config.yaml',
              help='Output file path')
@click.option('--profile', type=click.Choice(['default', 'dev', 'test', 'prod']),
              help='Initialize from profile')
@click.option('--preset', help='Initialize from preset')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
def init_config(format, output, profile, preset, interactive):
    """
    Initialize a new configuration file

    Create a new configuration file with default values or from a profile/preset.

    \b
    Examples:
      evolve config init --format yaml --output myconfig.yaml
      evolve config init --profile prod
      evolve config init --preset fast --output fast.config.yaml
      evolve config init --interactive
    """
    from ...unified.config import UnifiedEvolutionConfig, ConfigManager
    from ...unified.config_mapper import ConfigMapper

    try:
        if interactive:
            # Interactive mode
            click.echo("Initializing configuration in interactive mode...")
            click.echo("Press Enter for default values\n")

            # Collect values interactively
            mode = click.prompt('Evolution mode (pes/qd/mo/adversarial/standard/auto)',
                               default='auto', type=click.Choice(['pes', 'qd', 'mo', 'adversarial', 'standard', 'auto']))

            domain = click.prompt('Domain (general/finance/trading/science/engineering/pharma/web/math/ml)',
                                 default='general', type=click.Choice(['general', 'finance', 'trading', 'science',
                                                                        'engineering', 'pharma', 'web', 'math', 'ml']))

            max_iterations = click.prompt('Max iterations', default=100, type=int)
            population_size = click.prompt('Population size', default=50, type=int)

            # Create config with interactive values
            config_obj = UnifiedEvolutionConfig(
                mode=mode,
                domain=domain,
                max_iterations=max_iterations,
                population_size=population_size
            )

        elif preset:
            # Initialize from preset
            from ...unified.config import PresetManager
            preset_mgr = PresetManager()
            config_obj = preset_mgr.apply_preset(preset, UnifiedEvolutionConfig())
            click.echo(f"[OK] Applied preset '{preset}'")

        elif profile:
            # Initialize from profile
            from ...unified.config import ProfileManager
            profile_mgr = ProfileManager()
            config_obj = profile_mgr.load_profile(profile)
            click.echo(f"[OK] Loaded profile '{profile}'")

        else:
            # Default initialization
            config_obj = UnifiedEvolutionConfig()

        # Save configuration
        manager = ConfigManager()
        manager.save_config(config_obj, output, format)

        click.echo(f"[OK] Configuration initialized: {output}")
        click.echo(f"  Format: {format}")
        click.echo(f"  Edit this file to customize settings")

    except Exception as e:
        click.echo(f"[FAIL] Error initializing configuration: {e}", err=True)
        raise click.Abort()


@config.command('validate')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--format', type=click.Choice(['text', 'json', 'yaml']), default='text',
              help='Output format')
def validate_config(config_file, verbose, format):
    """
    Validate configuration file

    Check if configuration file is valid and show any errors.

    \b
    Examples:
      evolve config validate evolve.config.yaml
      evolve config validate myconfig.yaml --verbose
      evolve config validate myconfig.yaml --format json
    """
    from ...unified.config import ConfigManager, ConfigValidator

    try:
        manager = ConfigManager()
        config_obj = manager.load_config(config_file)

        validator = ConfigValidator()
        result = validator.validate(config_obj)

        if result.is_valid:
            output = {
                'status': 'valid',
                'file': config_file,
                'warnings': result.warnings if hasattr(result, 'warnings') else []
            }

            if format == 'json':
                click.echo(json.dumps(output, indent=2))

            elif format == 'yaml':
                click.echo(yaml.dump(output, default_flow_style=False))

            else:  # text
                click.echo("[OK] Configuration is valid")
                if output.get('warnings'):
                    click.echo("\nWarnings:")
                    for warning in output['warnings']:
                        click.echo(f"  ⚠ {warning}")

            return 0

        else:
            output = {
                'status': 'invalid',
                'file': config_file,
                'errors': result.errors
            }

            if format == 'json':
                click.echo(json.dumps(output, indent=2))

            elif format == 'yaml':
                click.echo(yaml.dump(output, default_flow_style=False))

            else:  # text
                click.echo(f"[FAIL] Configuration has {len(result.errors)} error(s):", err=True)
                for error in result.errors:
                    click.echo(f"  - {error}", err=True)

            return 1

    except Exception as e:
        click.echo(f"[FAIL] Error validating configuration: {e}", err=True)
        return 1


@config.command('list')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), default='table',
              help='Output format')
@click.option('--filter', help='Filter parameters by name')
@click.option('--category', help='Filter by category')
def list_params(format, filter, category):
    """
    List all configuration parameters

    Display all available configuration parameters with their types, defaults, and descriptions.

    \b
    Examples:
      evolve config list --format table
      evolve config list --filter iteration
      evolve config list --category llm --format json
    """
    from ...unified.config import get_all_parameters

    try:
        params = get_all_parameters()

        # Apply filters
        if filter:
            params = {k: v for k, v in params.items() if filter.lower() in k.lower()}

        if category:
            params = {k: v for k, v in params.items() if v.get('category') == category}

        if format == 'table':
            try:
                from tabulate import tabulate

                rows = []
                for name, info in sorted(params.items()):
                    desc = info.get('description', '')[:50]
                    if len(info.get('description', '')) > 50:
                        desc += '...'

                    rows.append([
                        name,
                        info.get('type', 'unknown'),
                        str(info.get('default', ''))[:30],
                        desc
                    ])

                click.echo(tabulate(rows, headers=['Parameter', 'Type', 'Default', 'Description'],
                                   tablefmt='grid'))
            except ImportError:
                click.echo("tabulate not installed. Install with: pip install tabulate")
                click.echo("\nParameters:")
                for name, info in sorted(params.items()):
                    click.echo(f"  {name}: {info.get('type', 'unknown')}")

        elif format == 'json':
            click.echo(json.dumps(params, indent=2))

        elif format == 'yaml':
            click.echo(yaml.dump(params, default_flow_style=False))

    except Exception as e:
        click.echo(f"[FAIL] Error listing parameters: {e}", err=True)
        raise click.Abort()


@config.command('get')
@click.argument('param_name')
@click.option('--config', type=click.Path(exists=True), help='Config file (uses default if not specified)')
@click.option('--format', type=click.Choice(['value', 'json', 'yaml']), default='value',
              help='Output format')
def get_param(param_name, config, format):
    """
    Get parameter value

    Retrieve the value of a specific parameter from the configuration.

    \b
    Examples:
      evolve config get max_iterations
      evolve config get mode --config myconfig.yaml
      evolve config get llm.models --format json
    """
    from ...unified.config import ConfigManager

    try:
        manager = ConfigManager()

        if config:
            cfg = manager.load_config(config_file=config)
        else:
            cfg = manager.load_config()

        # Navigate nested attributes
        value = _get_nested_attr(cfg, param_name)

        if value is not None:
            if format == 'json':
                click.echo(json.dumps(value, indent=2, default=str))
            elif format == 'yaml':
                click.echo(yaml.dump(value, default_flow_style=False))
            else:
                click.echo(f"{param_name}: {value}")

        else:
            click.echo(f"[FAIL] Parameter '{param_name}' not found", err=True)
            return 1

    except Exception as e:
        click.echo(f"[FAIL] Error getting parameter: {e}", err=True)
        return 1


@config.command('set')
@click.argument('param_name')
@click.argument('value')
@click.option('--config', type=click.Path(), help='Config file to modify (default: evolve.config.yaml)')
@click.option('--type', 'value_type', type=click.Choice(['auto', 'string', 'int', 'float', 'bool', 'json']),
              default='auto', help='Value type')
def set_param(param_name, value, config, value_type):
    """
    Set parameter value in config file

    Update a specific parameter in the configuration file.

    \b
    Examples:
      evolve config set max_iterations 200
      evolve config set enable_planning true
      evolve config set llm.temperature 0.8
      evolve config set llm.models '[{"name": "gpt-4"}]' --type json
    """
    from ...unified.config import ConfigManager

    try:
        manager = ConfigManager()

        # Determine config file
        config_file = config or 'evolve.config.yaml'

        # Load existing config
        if Path(config_file).exists():
            cfg = manager.load_config(config_file=config_file)
        else:
            click.echo(f"[FAIL] Config file '{config_file}' not found. Use 'evolve config init' first.", err=True)
            return 1

        # Parse value
        parsed_value = _parse_value(value, value_type)

        # Set parameter
        _set_nested_attr(cfg, param_name, parsed_value)

        # Save
        manager.save_config(cfg, config_file)

        click.echo(f"[OK] Set {param_name} = {parsed_value}")

    except Exception as e:
        click.echo(f"[FAIL] Error setting parameter: {e}", err=True)
        return 1


@config.command('diff')
@click.argument('config1', type=click.Path(exists=True))
@click.argument('config2', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['unified', 'table', 'json']), default='unified',
              help='Output format')
@click.option('--context', type=int, default=3, help='Number of context lines')
def diff_configs(config1, config2, format, context):
    """
    Compare two configuration files

    Show differences between two configuration files.

    \b
    Examples:
      evolve config diff config1.yaml config2.yaml
      evolve config diff base.yaml custom.yaml --format table
      evolve config diff old.yaml new.yaml --format json
    """
    from ...unified.config import ConfigManager

    try:
        manager = ConfigManager()
        cfg1 = manager.load_config(config_file=config1)
        cfg2 = manager.load_config(config_file=config2)

        if format == 'unified':
            # Unified diff format
            dict1 = cfg1.dict()
            dict2 = cfg2.dict()

            text1 = yaml.dump(dict1, default_flow_style=False, sort_keys=False)
            text2 = yaml.dump(dict2, default_flow_style=False, sort_keys=False)

            diff = unified_diff(
                text1.splitlines(keepends=True),
                text2.splitlines(keepends=True),
                fromfile=config1,
                tofile=config2,
                lineterm='',
                n=context
            )

            for line in diff:
                color = None
                if line.startswith('+') and not line.startswith('+++'):
                    color = 'green'
                elif line.startswith('-') and not line.startswith('---'):
                    color = 'red'

                if color:
                    click.secho(line, fg=color)
                else:
                    click.echo(line)

        elif format == 'json':
            diff = _compare_dicts(cfg1.dict(), cfg2.dict())
            click.echo(json.dumps(diff, indent=2))

        elif format == 'table':
            try:
                from tabulate import tabulate

                dict1 = cfg1.dict()
                dict2 = cfg2.dict()

                differences = _find_differences(dict1, dict2)

                if differences:
                    rows = []
                    for param, (val1, val2) in differences.items():
                        rows.append([param, str(val1), str(val2)])

                    click.echo(tabulate(rows, headers=['Parameter', config1, config2], tablefmt='grid'))
                else:
                    click.echo("No differences found")

            except ImportError:
                click.echo("tabulate not installed")

    except Exception as e:
        click.echo(f"[FAIL] Error comparing configs: {e}", err=True)
        return 1


@config.command('merge')
@click.argument('configs', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--output', type=click.Path(), required=True, help='Output file')
@click.option('--strategy', type=click.Choice(['override', 'merge', 'deep']), default='override',
              help='Merge strategy')
@click.option('--format', type=click.Choice(['yaml', 'json', 'toml']), default='yaml',
              help='Output format')
def merge_configs(configs, output, strategy, format):
    """
    Merge multiple configuration files

    Combine multiple configuration files into one.

    \b
    Examples:
      evolve config merge base.yaml domain.yaml overrides.yaml --output final.yaml
      evolve config merge *.yaml --output merged.yaml --strategy deep
    """
    from ...unified.config import ConfigManager

    try:
        if len(configs) < 2:
            click.echo("[FAIL] At least 2 config files required for merge", err=True)
            return 1

        manager = ConfigManager()
        merged = None

        for config_file in configs:
            cfg = manager.load_config(config_file=config_file)

            if merged is None:
                merged = cfg
            else:
                if strategy == 'override':
                    merged = _merge_override(merged, cfg)
                elif strategy == 'merge':
                    merged = _merge_shallow(merged, cfg)
                else:  # deep
                    merged = _merge_deep(merged, cfg)

        manager.save_config(merged, output, format)

        click.echo(f"[OK] Merged {len(configs)} configs -> {output}")
        click.echo(f"  Strategy: {strategy}")

    except Exception as e:
        click.echo(f"[FAIL] Error merging configs: {e}", err=True)
        return 1


@config.command('export')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), required=True)
@click.option('--format', type=click.Choice(['yaml', 'json', 'toml']), required=True)
def export_config(config_file, output, format):
    """
    Export configuration to different format

    Convert configuration file to different format.

    \b
    Examples:
      evolve config export evolve.config.yaml --output config.json --format json
      evolve config export config.yaml --output config.toml --format toml
    """
    from ...unified.config import ConfigManager

    try:
        manager = ConfigManager()
        config = manager.load_config(config_file=config_file)
        manager.save_config(config, output, format)

        click.echo(f"[OK] Exported {config_file} -> {output} (format: {format})")

    except Exception as e:
        click.echo(f"[FAIL] Error exporting config: {e}", err=True)
        return 1


@config.command('import')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), default='evolve.config.yaml')
@click.option('--validate', is_flag=True, help='Validate after import')
def import_config(config_file, output, validate):
    """
    Import configuration from file

    Import and optionally validate configuration file.

    \b
    Examples:
      evolve config import external_config.yaml
      evolve config import config.json --output myconfig.yaml --validate
    """
    from ...unified.config import ConfigManager, ConfigValidator

    try:
        manager = ConfigManager()
        config = manager.load_config(config_file=config_file)

        if validate:
            validator = ConfigValidator()
            result = validator.validate(config)

            if not result.is_valid:
                click.echo("[FAIL] Configuration validation failed:", err=True)
                for error in result.errors:
                    click.echo(f"  - {error}", err=True)
                return 1

        manager.save_config(config, output)

        click.echo(f"[OK] Imported {config_file} -> {output}")

    except Exception as e:
        click.echo(f"[FAIL] Error importing config: {e}", err=True)
        return 1


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _get_nested_attr(obj, attr_path):
    """Get nested attribute from object using dot notation"""
    attrs = attr_path.split('.')
    value = obj

    for attr in attrs:
        if hasattr(value, attr):
            value = getattr(value, attr)
        elif isinstance(value, dict) and attr in value:
            value = value[attr]
        else:
            return None

    return value


def _set_nested_attr(obj, attr_path, value):
    """Set nested attribute on object using dot notation"""
    attrs = attr_path.split('.')
    current = obj

    for attr in attrs[:-1]:
        if hasattr(current, attr):
            current = getattr(current, attr)
        elif isinstance(current, dict):
            if attr not in current:
                current[attr] = {}
            current = current[attr]

    final_attr = attrs[-1]

    if hasattr(current, final_attr):
        setattr(current, final_attr, value)
    elif isinstance(current, dict):
        current[final_attr] = value


def _parse_value(value, value_type):
    """Parse value string to appropriate type"""
    if value_type == 'string':
        return value

    elif value_type == 'int':
        return int(value)

    elif value_type == 'float':
        return float(value)

    elif value_type == 'bool':
        return value.lower() in ('true', '1', 'yes', 'on')

    elif value_type == 'json':
        return json.loads(value)

    else:  # auto
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try bool
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Default to string
        return value


def _compare_dicts(dict1, dict2):
    """Compare two dictionaries and return differences"""
    differences = {}

    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        if key not in dict1:
            differences[key] = {'added': dict2[key]}
        elif key not in dict2:
            differences[key] = {'removed': dict1[key]}
        elif dict1[key] != dict2[key]:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                nested_diff = _compare_dicts(dict1[key], dict2[key])
                if nested_diff:
                    differences[key] = nested_diff
            else:
                differences[key] = {
                    'old': dict1[key],
                    'new': dict2[key]
                }

    return differences


def _find_differences(dict1, dict2, prefix=''):
    """Find differences between two dictionaries (flat)"""
    differences = {}

    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        full_key = f"{prefix}.{key}" if prefix else key

        if key not in dict1:
            differences[full_key] = (None, dict2[key])
        elif key not in dict2:
            differences[full_key] = (dict1[key], None)
        elif dict1[key] != dict2[key]:
            differences[full_key] = (dict1[key], dict2[key])

    return differences


def _merge_override(base, override):
    """Merge configs with override strategy (last wins)"""
    from ...unified.config import UnifiedEvolutionConfig

    base_dict = base.dict()
    override_dict = override.dict()

    merged_dict = {**base_dict, **override_dict}

    return UnifiedEvolutionConfig(**merged_dict)


def _merge_shallow(base, override):
    """Merge configs with shallow merge"""
    from ...unified.config import UnifiedEvolutionConfig

    base_dict = base.dict()
    override_dict = override.dict()

    # Start with base, update with override (only top-level)
    merged_dict = base_dict.copy()
    for key, value in override_dict.items():
        if value is not None:  # Don't override with None
            merged_dict[key] = value

    return UnifiedEvolutionConfig(**merged_dict)


def _merge_deep(base, override):
    """Merge configs with deep merge"""
    from ...unified.config import UnifiedEvolutionConfig

    def _deep_merge(dict1, dict2):
        """Deep merge two dictionaries"""
        result = dict1.copy()

        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _deep_merge(result[key], value)
            elif value is not None:
                result[key] = value

        return result

    merged_dict = _deep_merge(base.dict(), override.dict())
    return UnifiedEvolutionConfig(**merged_dict)
