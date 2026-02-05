"""
Preset Management Commands

Provides commands for listing, applying, and managing configuration presets.

Commands:
    list    - List all available presets
    show    - Show preset details
    apply   - Apply preset and save to config file
    create  - Create custom preset
    delete  - Delete custom preset
    validate - Validate a preset
"""

import click
import json
import yaml
from pathlib import Path


@click.group()
def preset():
    """
    Preset management commands

    Manage configuration presets for common use cases.

    \b
    Examples:
      evolve preset list
      evolve preset show fast
      evolve preset apply fast --output fast.config.yaml
    """
    pass


@preset.command('list')
@click.option('--category', type=click.Choice(['all', 'performance', 'domain', 'use_case', 'system', 'problem']),
              default='all', help='Filter by category')
@click.option('--format', type=click.Choice(['text', 'json', 'yaml']), default='text',
              help='Output format')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
def list_presets(category, format, verbose):
    """
    List all available presets

    Display all available configuration presets.

    \b
    Examples:
      evolve preset list
      evolve preset list --category domain
      evolve preset list --format json --verbose
    """
    from ...unified.config import PresetManager

    try:
        manager = PresetManager()
        presets = manager.list_presets(category)

        if format == 'json':
            output = {
                'category': category,
                'presets': [
                    {
                        'name': p.name,
                        'description': p.description,
                        'category': p.category
                    }
                    for p in presets
                ]
            }
            click.echo(json.dumps(output, indent=2))

        elif format == 'yaml':
            output = {
                'category': category,
                'presets': [
                    {
                        'name': p.name,
                        'description': p.description,
                        'category': p.category
                    }
                    for p in presets
                ]
            }
            click.echo(yaml.dump(output, default_flow_style=False))

        else:  # text
            if presets:
                click.echo(f"Presets (category: {category}):")

                for p in presets:
                    click.echo(f"  - {p.name}: {p.description}")

                    if verbose:
                        click.echo(f"    Category: {p.category}")
                        # Show some key parameters
                        config = manager.apply_preset(p.name, manager.base_config)
                        click.echo(f"    Mode: {config.mode}")
                        click.echo(f"    Domain: {config.domain}")
                        click.echo(f"    Max iterations: {config.max_iterations}")
            else:
                click.echo(f"No presets found for category: {category}")

    except Exception as e:
        click.echo(f"[FAIL] Error listing presets: {e}", err=True)
        raise click.Abort()


@preset.command('show')
@click.argument('name')
@click.option('--format', type=click.Choice(['yaml', 'json', 'table']), default='yaml',
              help='Output format')
def show_preset(name, format):
    """
    Show preset details

    Display detailed information about a specific preset.

    \b
    Examples:
      evolve preset show fast
      evolve preset show balanced --format json
      evolve preset show thorough --format table
    """
    from ...unified.config import PresetManager, UnifiedEvolutionConfig

    try:
        manager = PresetManager()
        preset = manager.get_preset(name)

        # Apply preset to see the full configuration
        base_config = UnifiedEvolutionConfig()
        config = manager.apply_preset(name, base_config)

        click.echo(f"Preset: {preset.name}")
        click.echo(f"Description: {preset.description}")
        click.echo(f"Category: {preset.category}")

        if format == 'json':
            click.echo("\nConfiguration:")
            click.echo(json.dumps(config.dict(), indent=2, default=str))

        elif format == 'yaml':
            click.echo("\nConfiguration:")
            click.echo(yaml.dump(config.dict(), default_flow_style=False, sort_keys=False))

        elif format == 'table':
            try:
                from tabulate import tabulate

                rows = []
                for key, value in config.dict().items():
                    value_str = str(value)[:50]
                    if len(str(value)) > 50:
                        value_str += '...'
                    rows.append([key, value_str])

                click.echo("\nParameters:")
                click.echo(tabulate(rows, headers=['Parameter', 'Value'], tablefmt='grid'))

            except ImportError:
                click.echo("tabulate not installed. Use --format yaml instead")

    except Exception as e:
        click.echo(f"[FAIL] Error showing preset: {e}", err=True)
        return 1


@preset.command('apply')
@click.argument('name')
@click.option('--output', type=click.Path(), help='Output config file')
@click.option('--format', type=click.Choice(['yaml', 'json', 'toml']), default='yaml',
              help='Output format')
@click.option('--base', type=click.Path(exists=True), help='Base config file')
def apply_preset(name, output, format, base):
    """
    Apply preset and save to config file

    Apply a preset to create a configuration file.

    \b
    Examples:
      evolve preset apply fast --output fast.config.yaml
      evolve preset apply balanced --format json
      evolve preset apply thorough --base myconfig.yaml --output final.yaml
    """
    from ...unified.config import PresetManager, ConfigManager, UnifiedEvolutionConfig

    try:
        preset_mgr = PresetManager()
        config_mgr = ConfigManager()

        # Load base config
        if base:
            base_config = config_mgr.load_config(config_file=base)
        else:
            base_config = UnifiedEvolutionConfig()

        # Apply preset
        config = preset_mgr.apply_preset(name, base_config)

        # Determine output file
        output_file = output or f"{name}.config.{format}"

        # Save
        config_mgr.save_config(config, output_file, format)

        click.echo(f"[OK] Applied preset '{name}' to {output_file}")

    except Exception as e:
        click.echo(f"[FAIL] Error applying preset: {e}", err=True)
        return 1


@preset.command('create')
@click.argument('name')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--description', required=True, help='Preset description')
@click.option('--category', type=click.Choice(['performance', 'domain', 'use_case', 'system', 'problem']),
              required=True, help='Preset category')
def create_preset(name, config_file, description, category):
    """
    Create custom preset from config file

    Create a custom preset from an existing configuration file.

    \b
    Examples:
      evolve preset create mypreset myconfig.yaml --description "My custom preset" --category use_case
    """
    from ...unified.config import PresetManager, ConfigManager

    try:
        config_mgr = ConfigManager()
        preset_mgr = PresetManager()

        config = config_mgr.load_config(config_file=config_file)
        preset = preset_mgr.create_preset(name, config, description, category)

        click.echo(f"[OK] Created preset '{name}'")
        click.echo(f"  Category: {category}")
        click.echo(f"  Description: {description}")

    except Exception as e:
        click.echo(f"[FAIL] Error creating preset: {e}", err=True)
        return 1


@preset.command('delete')
@click.argument('name')
@click.option('--force', '-f', is_flag=True, help='Force deletion without confirmation')
def delete_preset(name, force):
    """
    Delete a custom preset

    Remove a custom preset from the presets directory.

    \b
    Examples:
      evolve preset delete mypreset
      evolve preset delete oldpreset --force
    """
    from ...unified.config import PresetManager

    try:
        # Check if it's a built-in preset
        manager = PresetManager()
        built_in_presets = ['fast', 'balanced', 'thorough', 'minimal', 'maximum']

        if name in built_in_presets:
            click.echo(f"[FAIL] Cannot delete built-in preset '{name}'", err=True)
            return 1

        if not force:
            if not click.confirm(f"Are you sure you want to delete preset '{name}'?"):
                click.echo("Aborted")
                return

        manager.delete_preset(name)

        click.echo(f"[OK] Deleted preset '{name}'")

    except Exception as e:
        click.echo(f"[FAIL] Error deleting preset: {e}", err=True)
        return 1


@preset.command('validate')
@click.argument('name')
def validate_preset(name):
    """
    Validate a preset

    Check if a preset is valid.

    \b
    Examples:
      evolve preset validate fast
      evolve preset validate mypreset
    """
    from ...unified.config import PresetManager, ConfigValidator, UnifiedEvolutionConfig

    try:
        preset_mgr = PresetManager()
        validator = ConfigValidator()

        # Apply preset to get config
        base_config = UnifiedEvolutionConfig()
        config = preset_mgr.apply_preset(name, base_config)

        result = validator.validate(config)

        if result.is_valid:
            click.echo(f"[OK] Preset '{name}' is valid")
            return 0
        else:
            click.echo(f"[FAIL] Preset '{name}' has {len(result.errors)} error(s):", err=True)
            for error in result.errors:
                click.echo(f"  - {error}", err=True)
            return 1

    except Exception as e:
        click.echo(f"[FAIL] Error validating preset: {e}", err=True)
        return 1


@preset.command('compare')
@click.argument('preset1')
@click.argument('preset2')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), default='table')
def compare_presets(preset1, preset2, format):
    """
    Compare two presets

    Show differences between two presets.

    \b
    Examples:
      evolve preset compare fast balanced
      evolve preset compare minimal maximum --format json
    """
    from ...unified.config import PresetManager, UnifiedEvolutionConfig

    try:
        manager = PresetManager()
        base_config = UnifiedEvolutionConfig()

        config1 = manager.apply_preset(preset1, base_config)
        config2 = manager.apply_preset(preset2, base_config)

        from ..commands.config import _find_differences
        differences = _find_differences(config1.dict(), config2.dict())

        if format == 'json':
            click.echo(json.dumps(differences, indent=2))

        elif format == 'yaml':
            click.echo(yaml.dump(differences, default_flow_style=False))

        elif format == 'table':
            try:
                from tabulate import tabulate

                if differences:
                    rows = []
                    for param, (val1, val2) in differences.items():
                        rows.append([param, str(val1)[:30], str(val2)[:30]])

                    click.echo(tabulate(rows, headers=['Parameter', preset1, preset2],
                                       tablefmt='grid'))
                else:
                    click.echo("No differences found")

            except ImportError:
                click.echo("tabulate not installed")

    except Exception as e:
        click.echo(f"[FAIL] Error comparing presets: {e}", err=True)
        return 1


@preset.command('categories')
def list_categories():
    """
    List all preset categories

    Display all available preset categories.

    \b
    Examples:
      evolve preset categories
    """
    categories = {
        'performance': 'Performance-focused presets (fast, balanced, thorough)',
        'domain': 'Domain-specific presets (finance, trading, science, etc.)',
        'use_case': 'Use-case specific presets (development, production, testing)',
        'system': 'System resource presets (minimal, standard, maximum)',
        'problem': 'Problem type presets (optimization, search, learning)'
    }

    click.echo("Preset Categories:")
    click.echo()

    for cat, description in categories.items():
        click.echo(f"  {cat}: {description}")


@preset.command('search')
@click.argument('query')
@click.option('--category', help='Search in specific category')
def search_presets(query, category):
    """
    Search presets by name or description

    Search for presets matching a query.

    \b
    Examples:
      evolve preset search fast
      evolve preset search optimization --category performance
    """
    from ...unified.config import PresetManager

    try:
        manager = PresetManager()
        presets = manager.list_presets(category or 'all')

        query_lower = query.lower()
        matching = []

        for preset in presets:
            if (query_lower in preset.name.lower() or
                query_lower in preset.description.lower()):
                matching.append(preset)

        if matching:
            click.echo(f"Found {len(matching)} preset(s) matching '{query}':")

            for preset in matching:
                click.echo(f"  - {preset.name}: {preset.description}")
                click.echo(f"    Category: {preset.category}")
        else:
            click.echo(f"No presets found matching '{query}'")

    except Exception as e:
        click.echo(f"[FAIL] Error searching presets: {e}", err=True)
        return 1
