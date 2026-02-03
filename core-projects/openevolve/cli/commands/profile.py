"""
Profile Management Commands

Provides commands for creating, listing, and managing configuration profiles.

Commands:
    create  - Create a new profile
    list    - List all available profiles
    show    - Show profile details
    apply   - Apply a profile to config file
    delete  - Delete a profile
    export  - Export profile to file
    import  - Import profile from file
    update  - Update existing profile
"""

import click
import json
import yaml
from pathlib import Path


@click.group()
def profile():
    """
    Profile management commands

    Manage configuration profiles for different environments and use cases.

    \b
    Examples:
      evolve profile list
      evolve profile create myprofile --base prod
      evolve profile apply dev
      evolve profile show prod
    """
    pass


@profile.command('create')
@click.argument('name')
@click.option('--base', type=click.Choice(['default', 'dev', 'test', 'prod']),
              default='default', help='Base profile to copy')
@click.option('--description', help='Profile description')
@click.option('--config', type=click.Path(exists=True),
              help='Create from existing config file')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
def create_profile(name, base, description, config, interactive):
    """
    Create a new configuration profile

    Create a custom profile from a base profile or existing config.

    \b
    Examples:
      evolve profile create myprofile --base prod --description "My custom profile"
      evolve profile create custom --config myconfig.yaml
      evolve profile create testprofile --interactive
    """
    from ...unified.config import ProfileManager, ConfigManager, UnifiedEvolutionConfig

    try:
        profile_mgr = ProfileManager()

        # Check if profile already exists
        existing_profiles = profile_mgr.list_profiles()
        if name in existing_profiles:
            if not click.confirm(f"Profile '{name}' already exists. Overwrite?"):
                click.echo("Aborted")
                return

        if config:
            # Create from existing config file
            config_mgr = ConfigManager()
            profile_config = config_mgr.load_config(config_file=config)
            click.echo(f"✓ Loaded config from {config}")

        elif interactive:
            # Interactive mode
            click.echo(f"Creating profile '{name}' in interactive mode...")
            click.echo("Press Enter for default values\n")

            mode = click.prompt('Evolution mode',
                               default='auto',
                               type=click.Choice(['pes', 'qd', 'mo', 'adversarial', 'standard', 'auto']))

            domain = click.prompt('Domain', default='general')

            max_iterations = click.prompt('Max iterations', default=100, type=int)
            population_size = click.prompt('Population size', default=50, type=int)

            profile_config = UnifiedEvolutionConfig(
                mode=mode,
                domain=domain,
                max_iterations=max_iterations,
                population_size=population_size
            )

        else:
            # Create from base profile
            base_config = profile_mgr.load_profile(base)
            profile_config = base_config
            click.echo(f"✓ Loaded base profile '{base}'")

        # Add description
        if description:
            if not hasattr(profile_config, 'description'):
                profile_config.description = description

        # Save profile
        profile_mgr.save_profile(name, profile_config, description)

        click.echo(f"✓ Profile '{name}' created")

    except Exception as e:
        click.echo(f"✗ Error creating profile: {e}", err=True)
        raise click.Abort()


@profile.command('list')
@click.option('--format', type=click.Choice(['text', 'json', 'yaml']), default='text',
              help='Output format')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
def list_profiles(format, verbose):
    """
    List all available profiles

    Display all available configuration profiles.

    \b
    Examples:
      evolve profile list
      evolve profile list --format json
      evolve profile list --verbose
    """
    from ...unified.config import ProfileManager

    try:
        manager = ProfileManager()
        profiles = manager.list_profiles()

        if format == 'json':
            output = {'profiles': profiles}
            click.echo(json.dumps(output, indent=2))

        elif format == 'yaml':
            output = {'profiles': profiles}
            click.echo(yaml.dump(output, default_flow_style=False))

        else:  # text
            if profiles:
                click.echo("Available profiles:")
                for profile_name in sorted(profiles):
                    click.echo(f"  - {profile_name}")

                    if verbose:
                        try:
                            profile_config = manager.load_profile(profile_name)
                            description = getattr(profile_config, 'description', 'No description')
                            click.echo(f"    Description: {description}")
                        except:
                            pass
            else:
                click.echo("No profiles found")

    except Exception as e:
        click.echo(f"✗ Error listing profiles: {e}", err=True)
        raise click.Abort()


@profile.command('show')
@click.argument('name')
@click.option('--format', type=click.Choice(['yaml', 'json', 'table']), default='yaml',
              help='Output format')
def show_profile(name, format):
    """
    Show profile details

    Display detailed information about a specific profile.

    \b
    Examples:
      evolve profile show prod
      evolve profile show dev --format json
      evolve profile show test --format table
    """
    from ...unified.config import ProfileManager

    try:
        manager = ProfileManager()
        profile_config = manager.load_profile(name)

        if format == 'json':
            click.echo(json.dumps(profile_config.dict(), indent=2, default=str))

        elif format == 'yaml':
            click.echo(yaml.dump(profile_config.dict(), default_flow_style=False, sort_keys=False))

        elif format == 'table':
            try:
                from tabulate import tabulate

                rows = []
                for key, value in profile_config.dict().items():
                    value_str = str(value)[:50]
                    if len(str(value)) > 50:
                        value_str += '...'
                    rows.append([key, value_str])

                click.echo(f"Profile: {name}")
                click.echo(tabulate(rows, headers=['Parameter', 'Value'], tablefmt='grid'))

            except ImportError:
                click.echo("tabulate not installed. Use --format yaml instead")

    except Exception as e:
        click.echo(f"✗ Error showing profile: {e}", err=True)
        return 1


@profile.command('apply')
@click.argument('name')
@click.option('--output', type=click.Path(), help='Output config file (default: evolve.config.yaml)')
@click.option('--format', type=click.Choice(['yaml', 'json', 'toml']), default='yaml',
              help='Output format')
def apply_profile(name, output, format):
    """
    Apply a profile to config file

    Load a profile and save it as a configuration file.

    \b
    Examples:
      evolve profile apply prod
      evolve profile apply dev --output dev.config.yaml
      evolve profile apply test --format json
    """
    from ...unified.config import ProfileManager, ConfigManager

    try:
        profile_mgr = ProfileManager()
        config_mgr = ConfigManager()

        # Load profile
        profile_config = profile_mgr.load_profile(name)

        # Save to config file
        output_file = output or 'evolve.config.yaml'
        config_mgr.save_config(profile_config, output_file, format)

        click.echo(f"✓ Applied profile '{name}' to {output_file}")

    except Exception as e:
        click.echo(f"✗ Error applying profile: {e}", err=True)
        return 1


@profile.command('delete')
@click.argument('name')
@click.option('--force', '-f', is_flag=True, help='Force deletion without confirmation')
def delete_profile(name, force):
    """
    Delete a profile

    Remove a profile from the profiles directory.

    \b
    Examples:
      evolve profile delete myprofile
      evolve profile delete oldprofile --force
    """
    from ...unified.config import ProfileManager

    try:
        # Prevent deletion of built-in profiles
        built_in = ['default', 'dev', 'test', 'prod']
        if name in built_in:
            click.echo(f"✗ Cannot delete built-in profile '{name}'", err=True)
            return 1

        if not force:
            if not click.confirm(f"Are you sure you want to delete profile '{name}'?"):
                click.echo("Aborted")
                return

        manager = ProfileManager()
        manager.delete_profile(name)

        click.echo(f"✓ Deleted profile '{name}'")

    except Exception as e:
        click.echo(f"✗ Error deleting profile: {e}", err=True)
        return 1


@profile.command('export')
@click.argument('name')
@click.option('--output', type=click.Path(), required=True, help='Output file path')
@click.option('--format', type=click.Choice(['yaml', 'json', 'toml']), default='yaml',
              help='Output format')
def export_profile(name, output, format):
    """
    Export profile to file

    Export a profile to a standalone configuration file.

    \b
    Examples:
      evolve profile export prod --output prod-backup.yaml
      evolve profile export dev --output dev.json --format json
    """
    from ...unified.config import ProfileManager, ConfigManager

    try:
        profile_mgr = ProfileManager()
        config_mgr = ConfigManager()

        profile_config = profile_mgr.load_profile(name)
        config_mgr.save_config(profile_config, output, format)

        click.echo(f"✓ Exported profile '{name}' to {output}")

    except Exception as e:
        click.echo(f"✗ Error exporting profile: {e}", err=True)
        return 1


@profile.command('import')
@click.argument('file', type=click.Path(exists=True))
@click.argument('name')
@click.option('--description', help='Profile description')
def import_profile(file, name, description):
    """
    Import profile from file

    Import a configuration file as a new profile.

    \b
    Examples:
      evolve profile import myconfig.yaml myprofile
      evolve profile import config.json myprofile --description "My custom profile"
    """
    from ...unified.config import ProfileManager, ConfigManager

    try:
        config_mgr = ConfigManager()
        profile_config = config_mgr.load_config(config_file=file)

        profile_mgr = ProfileManager()
        profile_mgr.save_profile(name, profile_config, description)

        click.echo(f"✓ Imported profile '{name}' from {file}")

    except Exception as e:
        click.echo(f"✗ Error importing profile: {e}", err=True)
        return 1


@profile.command('update')
@click.argument('name')
@click.option('--description', help='New description')
@click.option('--config', type=click.Path(exists=True), help='Update from config file')
def update_profile(name, description, config):
    """
    Update existing profile

    Update a profile's description or content.

    \b
    Examples:
      evolve profile update myprofile --description "Updated description"
      evolve profile update myprofile --config newconfig.yaml
    """
    from ...unified.config import ProfileManager, ConfigManager

    try:
        profile_mgr = ProfileManager()
        config_mgr = ConfigManager()

        if config:
            # Update from config file
            new_config = config_mgr.load_config(config_file=config)
            profile_mgr.save_profile(name, new_config, description)
            click.echo(f"✓ Updated profile '{name}' from {config}")
        elif description:
            # Update description only
            profile_config = profile_mgr.load_profile(name)
            profile_mgr.save_profile(name, profile_config, description)
            click.echo(f"✓ Updated profile '{name}' description")
        else:
            click.echo("✗ No changes specified. Use --description or --config", err=True)
            return 1

    except Exception as e:
        click.echo(f"✗ Error updating profile: {e}", err=True)
        return 1


@profile.command('validate')
@click.argument('name')
def validate_profile(name):
    """
    Validate a profile

    Check if a profile is valid.

    \b
    Examples:
      evolve profile validate prod
      evolve profile validate myprofile
    """
    from ...unified.config import ProfileManager, ConfigValidator

    try:
        profile_mgr = ProfileManager()
        validator = ConfigValidator()

        profile_config = profile_mgr.load_profile(name)
        result = validator.validate(profile_config)

        if result.is_valid:
            click.echo(f"✓ Profile '{name}' is valid")
            return 0
        else:
            click.echo(f"✗ Profile '{name}' has {len(result.errors)} error(s):", err=True)
            for error in result.errors:
                click.echo(f"  - {error}", err=True)
            return 1

    except Exception as e:
        click.echo(f"✗ Error validating profile: {e}", err=True)
        return 1


@profile.command('diff')
@click.argument('profile1')
@click.argument('profile2')
@click.option('--format', type=click.Choice(['unified', 'json', 'table']), default='unified')
def diff_profiles(profile1, profile2, format):
    """
    Compare two profiles

    Show differences between two profiles.

    \b
    Examples:
      evolve profile diff dev prod
      evolve profile diff default custom --format json
    """
    from ...unified.config import ProfileManager

    try:
        manager = ProfileManager()
        config1 = manager.load_profile(profile1)
        config2 = manager.load_profile(profile2)

        if format == 'json':
            from ..commands.config import _compare_dicts
            diff = _compare_dicts(config1.dict(), config2.dict())
            click.echo(json.dumps(diff, indent=2))

        elif format == 'unified':
            import difflib

            dict1 = config1.dict()
            dict2 = config2.dict()

            text1 = yaml.dump(dict1, default_flow_style=False, sort_keys=False)
            text2 = yaml.dump(dict2, default_flow_style=False, sort_keys=False)

            diff = difflib.unified_diff(
                text1.splitlines(keepends=True),
                text2.splitlines(keepends=True),
                fromfile=profile1,
                tofile=profile2,
                lineterm=''
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

        elif format == 'table':
            try:
                from tabulate import tabulate
                from ..commands.config import _find_differences

                differences = _find_differences(config1.dict(), config2.dict())

                if differences:
                    rows = []
                    for param, (val1, val2) in differences.items():
                        rows.append([param, str(val1)[:30], str(val2)[:30]])

                    click.echo(tabulate(rows, headers=['Parameter', profile1, profile2],
                                       tablefmt='grid'))
                else:
                    click.echo("No differences found")

            except ImportError:
                click.echo("tabulate not installed")

    except Exception as e:
        click.echo(f"✗ Error comparing profiles: {e}", err=True)
        return 1
