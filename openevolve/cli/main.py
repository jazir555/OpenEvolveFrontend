"""
OpenEvolve CLI - Main Entry Point

Provides 20+ commands for configuration, profile, preset, and environment management.

Usage:
    evolve [COMMAND] [OPTIONS]

Example:
    evolve config init --format yaml
    evolve profile list
    evolve preset apply fast
"""

import click
import sys
from pathlib import Path

# Import command groups
from .commands.config import config
from .commands.profile import profile
from .commands.preset import preset
from .commands.env import env
from .commands.validate import validate


@click.group()
@click.version_option(version="1.0.0", prog_name="evolve")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', type=click.Path(), help='Specify config file')
@click.pass_context
def evolve(ctx, verbose, config):
    """
    Unified Evolution Engine CLI

    Manage configurations, profiles, presets, and more.

    \b
    Common Commands:
      evolve config init      Initialize a new configuration
      evolve config validate  Validate configuration file
      evolve profile list     List available profiles
      evolve preset apply     Apply a preset
      evolve env list         List environment variables

    \b
    Getting Help:
      evolve --help           Show general help
      evolve [COMMAND] --help Show command-specific help

    \b
    Examples:
      evolve config init --format yaml --output myconfig.yaml
      evolve config validate evolve.config.yaml
      evolve profile apply prod
      evolve preset show fast
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Store verbose flag and config path in context
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config

    # Set up logging if verbose
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')


# Add command groups
evolve.add_command(config)
evolve.add_command(profile)
evolve.add_command(preset)
evolve.add_command(env)
evolve.add_command(validate)


# ============================================================================
# UTILITY COMMANDS
# ============================================================================

@evolve.command()
@click.option('--format', type=click.Choice(['json', 'yaml', 'text']), default='text')
def info(format):
    """
    Show system information

    Display version, configuration paths, and system status.

    Example:
        evolve info --format json
    """
    import platform
    import os
    from .. import __version__

    info_data = {
        'version': __version__,
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'architecture': platform.machine(),
        'config_path': _find_config_file(),
        'profiles_path': _get_profiles_path(),
        'presets_path': _get_presets_path(),
    }

    if format == 'json':
        import json
        click.echo(json.dumps(info_data, indent=2))

    elif format == 'yaml':
        import yaml
        click.echo(yaml.dump(info_data, default_flow_style=False))

    else:  # text
        click.echo("OpenEvolve CLI Information")
        click.echo("=" * 50)
        click.echo(f"Version: {info_data['version']}")
        click.echo(f"Python: {info_data['python_version']}")
        click.echo(f"Platform: {info_data['platform']} {info_data['architecture']}")
        click.echo(f"Config: {info_data['config_path']}")
        click.echo(f"Profiles: {info_data['profiles_path']}")
        click.echo(f"Presets: {info_data['presets_path']}")


@evolve.command()
@click.argument('query', required=False)
@click.option('--category', type=click.Choice(['all', 'config', 'profile', 'preset', 'env', 'validate']))
@click.option('--format', type=click.Choice(['text', 'json']), default='text')
def help_docs(query, category, format):
    """
    Show documentation (alias for --help)

    Search for help on specific topics.

    Example:
        evolve help_docs config init
        evolve help_docs --category profile
    """
    if query:
        # Show help for specific command
        try:
            ctx = evolve.make_context('evolve', ['--help'])
            click.echo_via_pager(ctx.get_help())
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    else:
        # Show general help
        click.echo(evolve.get_help(evolve.make_context('evolve', [])))


@evolve.command()
def completion():
    """
    Generate shell completion script

    Output shell completion code for bash or zsh.

    Bash:
        eval "$(evolve completion --bash)"

    Zsh:
        eval "$(evolve completion --zsh)"
    """
    shell = click.get_appdirectory()
    click.echo(f"# Shell completion for evolve")
    click.echo(f"# Add to your ~/.bashrc or ~/.zshrc:")
    click.echo()
    click.echo("# Bash:")
    click.echo("eval \"$(evolve completion --bash)\"")
    click.echo()
    click.echo("# Zsh:")
    click.echo("eval \"$(evolve completion --zsh)\"")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _find_config_file() -> str:
    """Find default config file"""
    candidates = [
        'evolve.config.yaml',
        'evolve.config.yml',
        'evolve.config.json',
        'evolve.config.toml',
    ]

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path.absolute())

    return "No config file found"


def _get_profiles_path() -> str:
    """Get profiles directory path"""
    from pathlib import Path
    module_path = Path(__file__).parent.parent
    profiles_path = module_path / "configs" / "profiles"
    return str(profiles_path.absolute()) if profiles_path.exists() else "Not found"


def _get_presets_path() -> str:
    """Get presets directory path"""
    from pathlib import Path
    module_path = Path(__file__).parent.parent
    presets_path = module_path / "configs" / "presets"
    return str(presets_path.absolute()) if presets_path.exists() else "Not found"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for evolve CLI"""
    evolve(obj={})


if __name__ == '__main__':
    main()
