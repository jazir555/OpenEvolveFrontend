"""
Environment Variable Management Commands

Provides commands for setting, listing, and managing environment variables.

Commands:
    set     - Set environment variable
    get     - Get environment variable value
    list    - List all evolution environment variables
    export  - Export environment variables to .env file
    load    - Load environment variables from .env file
    unset   - Unset environment variable
    validate - Validate environment variables
"""

import click
import os
from pathlib import Path


@click.group()
def env():
    """
    Environment variable management commands

    Manage environment variables for the Unified Evolution Engine.

    \b
    Examples:
      evolve env set MAX_ITERATIONS 200
      evolve env list
      evolve env export --output .env
    """
    pass


@env.command('set')
@click.argument('variable')
@click.argument('value')
@click.option('--export', type=click.Path(), help='Append to .env file')
@click.option('--permanent', is_flag=True, help='Set permanently in shell config')
def set_env(variable, value, export, permanent):
    """
    Set environment variable

    Set an environment variable for the current session.

    \b
    Examples:
      evolve env set MAX_ITERATIONS 200
      evolve env set ENABLE_PLANNING true
      evolve env set API_KEY secret123 --export .env
    """
    # Convert to uppercase with EVOLVE_ prefix if not present
    if not variable.startswith('EVOLVE_'):
        env_var = f"EVOLVE_{variable.upper()}"
    else:
        env_var = variable

    # Set in current environment
    os.environ[env_var] = value

    click.echo(f"[OK] Set {env_var}={value}")

    # Export to .env file if specified
    if export:
        env_path = Path(export)

        # Create file if it doesn't exist
        if not env_path.exists():
            env_path.touch()

        # Append to file
        with open(env_path, 'a') as f:
            f.write(f"{env_var}={value}\n")

        click.echo(f"  Exported to {export}")

    # Permanent (shell config) - provide instructions
    if permanent:
        click.echo("\nTo make this change permanent, add to your shell profile:")
        click.echo(f"  export {env_var}={value}")

        shell_rc = _detect_shell_rc()
        if shell_rc:
            click.echo(f"\nAdd this line to {shell_rc}:")
            click.echo(f"  echo 'export {env_var}={value}' >> {shell_rc}")


@env.command('get')
@click.argument('variable')
def get_env(variable):
    """
    Get environment variable value

    Display the value of an environment variable.

    \b
    Examples:
      evolve env get MAX_ITERATIONS
      evolve env get EVOLVE_MODE
    """
    # Convert to uppercase with EVOLVE_ prefix if not present
    if not variable.startswith('EVOLVE_'):
        env_var = f"EVOLVE_{variable.upper()}"
    else:
        env_var = variable

    value = os.environ.get(env_var)

    if value is not None:
        click.echo(f"{env_var}={value}")
    else:
        click.echo(f"[FAIL] Environment variable '{env_var}' not set", err=True)
        return 1


@env.command('list')
@click.option('--prefix', default='EVOLVE_', help='Filter by prefix')
@click.option('--format', type=click.Choice(['text', 'json', 'export']), default='text',
              help='Output format')
def list_env(prefix, format):
    """
    List all evolution environment variables

    Display all environment variables matching the prefix.

    \b
    Examples:
      evolve env list
      evolve env list --format json
      evolve env list --format export > .env
    """
    env_vars = {k: v for k, v in os.environ.items() if k.startswith(prefix)}

    if not env_vars:
        click.echo(f"No environment variables found with prefix '{prefix}'")
        return

    if format == 'json':
        import json
        click.echo(json.dumps(env_vars, indent=2))

    elif format == 'export':
        # Export format (VAR=value)
        for var, value in sorted(env_vars.items()):
            click.echo(f"{var}={value}")

    else:  # text
        click.echo(f"Environment variables (prefix: {prefix}):")

        for var, value in sorted(env_vars.items()):
            # Mask sensitive values
            if any(sensitive in var.upper() for sensitive in ['KEY', 'SECRET', 'PASSWORD', 'TOKEN']):
                value = '*' * len(value)

            click.echo(f"  {var}={value}")


@env.command('export')
@click.option('--output', type=click.Path(), default='.env', help='Output file path')
@click.option('--prefix', default='EVOLVE_', help='Filter by prefix')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing file')
def export_env(output, prefix, force):
    """
    Export environment variables to .env file

    Save all environment variables to a .env file.

    \b
    Examples:
      evolve env export
      evolve env export --output production.env
      evolve env export --force
    """
    env_vars = {k: v for k, v in os.environ.items() if k.startswith(prefix)}

    output_path = Path(output)

    # Check if file exists
    if output_path.exists() and not force:
        if not click.confirm(f"File '{output}' already exists. Overwrite?"):
            click.echo("Aborted")
            return

    # Write to file
    with open(output_path, 'w') as f:
        for var, value in sorted(env_vars.items()):
            # Quote values with spaces
            if ' ' in value:
                value = f'"{value}"'
            f.write(f"{var}={value}\n")

    click.echo(f"[OK] Exported {len(env_vars)} variables to {output}")


@env.command('load')
@click.argument('file', type=click.Path(exists=True))
@click.option('--dry-run', is_flag=True, help='Show what would be loaded without setting')
def load_env(file, dry_run):
    """
    Load environment variables from .env file

    Load environment variables from a file.

    \b
    Examples:
      evolve env load .env
      evolve env load production.env --dry-run
    """
    env_path = Path(file)
    count = 0

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse VAR=value
            if '=' in line:
                var, value = line.split('=', 1)
                var = var.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                if dry_run:
                    click.echo(f"  Would set: {var}={value}")
                else:
                    os.environ[var] = value
                    count += 1

    if dry_run:
        click.echo(f"\nDry run: Found variables in {file}")
    else:
        click.echo(f"[OK] Loaded {count} variable(s) from {file}")


@env.command('unset')
@click.argument('variable')
@click.option('--remove-from', type=click.Path(exists=True),
              help='Remove from .env file')
def unset_env(variable, remove_from):
    """
    Unset environment variable

    Remove an environment variable from the current session.

    \b
    Examples:
      evolve env unset MAX_ITERATIONS
      evolve env unset API_KEY --remove-from .env
    """
    # Convert to uppercase with EVOLVE_ prefix if not present
    if not variable.startswith('EVOLVE_'):
        env_var = f"EVOLVE_{variable.upper()}"
    else:
        env_var = variable

    # Unset from current environment
    if env_var in os.environ:
        del os.environ[env_var]
        click.echo(f"[OK] Unset {env_var}")
    else:
        click.echo(f"Environment variable '{env_var}' was not set")
        return 1

    # Remove from .env file if specified
    if remove_from:
        env_path = Path(remove_from)
        lines = []

        with open(env_path, 'r') as f:
            for line in f:
                # Skip lines that set this variable
                if not line.strip().startswith(f"{env_var}="):
                    lines.append(line)

        # Write back
        with open(env_path, 'w') as f:
            f.writelines(lines)

        click.echo(f"  Removed from {remove_from}")


@env.command('validate')
def validate_env():
    """
    Validate environment variables

    Check if all required environment variables are set and valid.

    \b
    Examples:
      evolve env validate
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


@env.command('show-config')
def show_env_config():
    """
    Show current environment configuration

    Display how environment variables are currently configured.

    \b
    Examples:
      evolve env show-config
    """
    from ...unified.config import EnvConfigParser

    try:
        parser = EnvConfigParser()
        config = parser.parse_env_config()

        import yaml
        click.echo("Current environment configuration:")
        click.echo(yaml.dump(config.dict(), default_flow_style=False, sort_keys=False))

    except Exception as e:
        click.echo(f"[FAIL] Error showing configuration: {e}", err=True)
        return 1


@env.command('template')
@click.option('--output', type=click.Path(), default='.env.template',
              help='Output template file')
@click.option('--with-examples', is_flag=True, help='Include example values')
def create_template(output, with_examples):
    """
    Create .env template file

    Generate a template .env file with all available environment variables.

    \b
    Examples:
      evolve env template
      evolve env template --output .env.example --with-examples
    """
    template_vars = {
        'EVOLVE_MODE': 'Auto-detect evolution mode (pes/qd/mo/adversarial/standard/auto)',
        'EVOLVE_DOMAIN': 'Problem domain (general/finance/trading/science/etc.)',
        'EVOLVE_MAX_ITERATIONS': 'Maximum number of iterations (default: 100)',
        'EVOLVE_POPULATION_SIZE': 'Population size (default: 50)',
        'EVOLVE_LLM_MODELS': 'LLM models (JSON array)',
        'EVOLVE_LLM_API_KEY': 'LLM API key',
        'EVOLVE_LLM_TEMPERATURE': 'LLM temperature (default: 0.7)',
        'EVOLVE_DATABASE_URL': 'Database connection URL',
        'EVOLVE_DATABASE_TYPE': 'Database type (sqlite/postgres/mysql)',
        'EVOLVE_LOG_LEVEL': 'Logging level (DEBUG/INFO/WARNING/ERROR)',
        'EVOLVE_TIMEOUT': 'Request timeout in seconds (default: 60)',
        'EVOLVE_ENABLE_PLANNING': 'Enable planning phase (true/false)',
        'EVOLVE_ENABLE_SUMMARY': 'Enable summary phase (true/false)',
    }

    output_path = Path(output)

    with open(output_path, 'w') as f:
        f.write("# OpenEvolve Environment Variables Template\n")
        f.write("# Copy this file to .env and fill in your values\n\n")

        for var, description in template_vars.items():
            f.write(f"# {description}\n")
            if with_examples:
                # Add example values
                if 'API_KEY' in var:
                    f.write(f"{var}=your_api_key_here\n")
                elif 'MODE' in var:
                    f.write(f"{var}=auto\n")
                elif 'ITERATIONS' in var:
                    f.write(f"{var}=100\n")
                elif 'POPULATION' in var:
                    f.write(f"{var}=50\n")
                elif 'TEMPERATURE' in var:
                    f.write(f"{var}=0.7\n")
                elif 'ENABLE' in var or 'LOG_LEVEL' in var:
                    f.write(f"{var}=INFO\n")
                else:
                    f.write(f"{var}=\n")
            else:
                f.write(f"{var}=\n")
            f.write("\n")

    click.echo(f"[OK] Created template file: {output}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _detect_shell_rc():
    """Detect shell configuration file"""
    import platform

    home = Path.home()

    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        # Check for bash or zsh
        if (home / '.zshrc').exists():
            return '~/.zshrc'
        elif (home / '.bashrc').exists():
            return '~/.bashrc'
        elif (home / '.bash_profile').exists():
            return '~/.bash_profile'
    elif platform.system() == 'Windows':
        # Windows doesn't typically use shell rc files
        return None

    return None
