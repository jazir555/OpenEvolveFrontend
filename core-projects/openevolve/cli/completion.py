"""
Shell Auto-completion Scripts

Provides bash and zsh completion support for the evolve CLI.
"""

# ============================================================================
# BASH COMPLETION
# ============================================================================

_BASH_COMPLETION = """
# Bash completion for evolve CLI

_evolve_completion() {
    local cur prev words cword
    _init_completion || return

    # Main commands
    local commands="config profile preset env validate help info completion"

    # Config subcommands
    local config_commands="init validate list get set diff merge export import"
    local config_formats="yaml json toml"
    local config_output_formats="table json yaml unified"

    # Profile subcommands
    local profile_commands="create list show apply delete export import update validate diff"
    local profile_builtins="default dev test prod"

    # Preset subcommands
    local preset_commands="list show apply create delete validate compare categories search"
    local preset_categories="all performance domain use_case system problem"

    # Validate subcommands
    local validate_commands="all config profile preset env check-all quick schema"

    # Environment subcommands
    local env_commands="set get list export load unset validate show-config template"

    # Complete main command
    if [[ ${cword} -eq 1 ]]; then
        COMPREPLY=($(compgen -W "${commands}" -- "${cur}"))
        return 0
    fi

    # Complete subcommands
    case ${prev} in
        config)
            COMPREPLY=($(compgen -W "${config_commands}" -- "${cur}"))
            return 0
            ;;
        profile)
            COMPREPLY=($(compgen -W "${profile_commands}" -- "${cur}"))
            return 0
            ;;
        preset)
            COMPREPLY=($(compgen -W "${preset_commands}" -- "${cur}"))
            return 0
            ;;
        env)
            COMPREPLY=($(compgen -W "${env_commands}" -- "${cur}"))
            return 0
            ;;
        validate)
            COMPREPLY=($(compgen -W "${validate_commands}" -- "${cur}"))
            return 0
            ;;
    esac

    # Complete options
    case "${words[1]}" in
        config)
            case "${words[2]}" in
                init)
                    case "${words[$((cword-1))]}" in
                        --format)
                            COMPREPLY=($(compgen -W "${config_formats}" -- "${cur}"))
                            return 0
                            ;;
                        --profile)
                            COMPREPLY=($(compgen -W "${profile_builtins}" -- "${cur}"))
                            return 0
                            ;;
                    esac
                    COMPREPLY=($(compgen -W "--format --output --profile --preset --interactive" -- "${cur}"))
                    ;;
                validate)
                    COMPREPLY=($(compgen -W "--verbose --format" -- "${cur}"))
                    _filedir
                    ;;
                list)
                    COMPREPLY=($(compgen -W "--format --filter --category" -- "${cur}"))
                    ;;
                get|set)
                    COMPREPLY=($(compgen -W "--config --format --type" -- "${cur}"))
                    ;;
                diff)
                    COMPREPLY=($(compgen -W "--format --context" -- "${cur}"))
                    _filedir
                    ;;
                merge)
                    COMPREPLY=($(compgen -W "--output --strategy --format" -- "${cur}"))
                    _filedir
                    ;;
                export)
                    COMPREPLY=($(compgen -W "--output --format" -- "${cur}"))
                    _filedir
                    ;;
                import)
                    COMPREPLY=($(compgen -W "--output --validate" -- "${cur}"))
                    _filedir
                    ;;
            esac
            ;;
        profile)
            case "${words[2]}" in
                create)
                    case "${words[$((cword-1))]}" in
                        --base)
                            COMPREPLY=($(compgen -W "${profile_builtins}" -- "${cur}"))
                            return 0
                            ;;
                    esac
                    COMPREPLY=($(compgen -W "--base --description --config --interactive" -- "${cur}"))
                    ;;
                show|apply|delete|export|validate)
                    # List available profiles
                    local profiles=$(evolve profile list 2>/dev/null | grep -E "^\\s+-" | sed 's/.*- //')
                    COMPREPLY=($(compgen -W "${profiles}" -- "${cur}"))
                    ;;
                import)
                    _filedir
                    ;;
                diff)
                    # List available profiles
                    local profiles=$(evolve profile list 2>/dev/null | grep -E "^\\s+-" | sed 's/.*- //')
                    COMPREPLY=($(compgen -W "${profiles}" -- "${cur}"))
                    ;;
            esac
            ;;
        preset)
            case "${words[2]}" in
                list)
                    COMPREPLY=($(compgen -W "--category --format --verbose" -- "${cur}"))
                    ;;
                show|apply|delete|validate)
                    # List available presets
                    local presets=$(evolve preset list 2>/dev/null | grep -E "^\\s+-" | sed 's/.*- \\([^:]*\\):.*/\\1/')
                    COMPREPLY=($(compgen -W "${presets}" -- "${cur}"))
                    ;;
                create)
                    COMPREPLY=($(compgen -W "--description --category" -- "${cur}"))
                    _filedir
                    ;;
                compare)
                    # List available presets
                    local presets=$(evolve preset list 2>/dev/null | grep -E "^\\s+-" | sed 's/.*- \\([^:]*\\):.*/\\1/')
                    COMPREPLY=($(compgen -W "${presets}" -- "${cur}"))
                    ;;
                search)
                    COMPREPLY=($(compgen -W "--category" -- "${cur}"))
                    ;;
            esac
            ;;
        env)
            case "${words[2]}" in
                set)
                    COMPREPLY=($(compgen -W "--export --permanent" -- "${cur}"))
                    ;;
                list)
                    COMPREPLY=($(compgen -W "--prefix --format" -- "${cur}"))
                    ;;
                export)
                    COMPREPLY=($(compgen -W "--output --prefix --force" -- "${cur}"))
                    ;;
                load|unset)
                    _filedir
                    ;;
            esac
            ;;
        validate)
            case "${words[2]}" in
                all)
                    COMPREPLY=($(compgen -W "--config --verbose --strict" -- "${cur}"))
                    ;;
                config|quick)
                    _filedir
                    ;;
                profile)
                    # List available profiles
                    local profiles=$(evolve profile list 2>/dev/null | grep -E "^\\s+-" | sed 's/.*- //')
                    COMPREPLY=($(compgen -W "${profiles}" -- "${cur}"))
                    ;;
                preset)
                    # List available presets
                    local presets=$(evolve preset list 2>/dev/null | grep -E "^\\s+-" | sed 's/.*- \\([^:]*\\):.*/\\1/')
                    COMPREPLY=($(compgen -W "${presets}" -- "${cur}"))
                    ;;
            esac
            ;;
    esac

    # File completion
    _filedir
}

complete -F _evolve_completion evolve
"""


# ============================================================================
# ZSH COMPLETION
# ============================================================================

_ZSH_COMPLETION = """
#compdef evolve

# Zsh completion for evolve CLI

_evolve() {
    local -a commands

    # Main commands
    commands=(
        'config:Configuration management commands'
        'profile:Profile management commands'
        'preset:Preset management commands'
        'env:Environment variable management commands'
        'validate:Validation commands'
        'help:Show help documentation'
        'info:Show system information'
        'completion:Generate shell completion script'
    )

    # Config subcommands
    local -a config_commands
    config_commands=(
        'init:Initialize a new configuration file'
        'validate:Validate configuration file'
        'list:List all configuration parameters'
        'get:Get parameter value'
        'set:Set parameter value'
        'diff:Compare two configuration files'
        'merge:Merge multiple configuration files'
        'export:Export configuration to different format'
        'import:Import configuration from file'
    )

    # Profile subcommands
    local -a profile_commands
    profile_commands=(
        'create:Create a new profile'
        'list:List all available profiles'
        'show:Show profile details'
        'apply:Apply a profile to config file'
        'delete:Delete a profile'
        'export:Export profile to file'
        'import:Import profile from file'
        'update:Update existing profile'
        'validate:Validate a profile'
        'diff:Compare two profiles'
    )

    # Preset subcommands
    local -a preset_commands
    preset_commands=(
        'list:List all available presets'
        'show:Show preset details'
        'apply:Apply preset and save to config file'
        'create:Create custom preset'
        'delete:Delete custom preset'
        'validate:Validate a preset'
        'compare:Compare two presets'
        'categories:List all preset categories'
        'search:Search presets by name or description'
    )

    # Environment subcommands
    local -a env_commands
    env_commands=(
        'set:Set environment variable'
        'get:Get environment variable value'
        'list:List all evolution environment variables'
        'export:Export environment variables to .env file'
        'load:Load environment variables from .env file'
        'unset:Unset environment variable'
        'validate:Validate environment variables'
        'show-config:Show current environment configuration'
        'template:Create .env template file'
    )

    # Validate subcommands
    local -a validate_commands
    validate_commands=(
        'all:Validate current configuration'
        'config:Validate specific config file'
        'profile:Validate a profile'
        'preset:Validate a preset'
        'env:Validate environment variables'
        'check-all:Run all validation checks'
        'quick:Quick validation check'
        'schema:Show configuration schema'
    )

    # Format options
    local -a format_options
    format_options=('yaml' 'json' 'toml')

    # Output format options
    local -a output_format_options
    output_format_options=('table' 'json' 'yaml' 'text' 'unified')

    # Profile options
    local -a profile_options
    profile_options=('default' 'dev' 'test' 'prod')

    # Preset categories
    local -a preset_categories
    preset_categories=('all' 'performance' 'domain' 'use_case' 'system' 'problem')

    local curcontext="$curcontext" state line
    typeset -A opt_args

    _arguments -C \\
        '(- *)--version[Show version and exit]' \\
        '(- *){-h,--help}'[Show help] \\
        '(-v --verbose)'{-v,--verbose}'[Enable verbose output]' \\
        '--config[Specify config file]:file:_files' \\
        '1: :->command' \\
        '*::arg:->args'

    case $state in
        command)
            _describe 'command' commands
            ;;
        args)
            case $words[2] in
                config)
                    _describe 'config command' config_commands
                    case $words[3] in
                        init)
                            _arguments \\
                                '--format[Output format]:format:($format_options)' \\
                                '--output[Output file path]:file:_files' \\
                                '--profile[Initialize from profile]:profile:($profile_options)' \\
                                '--preset[Initialize from preset]:preset:' \\
                                '(-i --interactive)'{-i,--interactive}'[Interactive mode]'
                            ;;
                        validate)
                            _arguments \\
                                '--verbose[Verbose output]' \\
                                '--format[Output format]:format:($output_format_options)' \\
                                '*:file:_files'
                            ;;
                        list)
                            _arguments \\
                                '--format[Output format]:format:($output_format_options)' \\
                                '--filter[Filter parameters by name]' \\
                                '--category[Filter by category]'
                            ;;
                        get)
                            _arguments \\
                                '--config[Config file]:file:_files' \\
                                '--format[Output format]:format:(value json yaml)' \\
                                ':param_name'
                            ;;
                        set)
                            _arguments \\
                                '--config[Config file]:file:_files' \\
                                '--type[Value type]:type:(auto string int float bool json)' \\
                                ':param_name' \\
                                ':value'
                            ;;
                        diff)
                            _arguments \\
                                '--format[Output format]:format:(unified table json)' \\
                                '--context[Number of context lines]:number:' \\
                                '*:file:_files'
                            ;;
                        merge)
                            _arguments \\
                                '--output[Output file]:file:_files' \\
                                '--strategy[Merge strategy]:strategy:(override merge deep)' \\
                                '--format[Output format]:format:($format_options)' \\
                                '*:file:_files'
                            ;;
                    esac
                    ;;
                profile)
                    _describe 'profile command' profile_commands
                    case $words[3] in
                        create)
                            _arguments \\
                                '--base[Base profile]:profile:($profile_options)' \\
                                '--description[Profile description]' \\
                                '--config[Create from existing config]:file:_files' \\
                                '(-i --interactive)'{-i,--interactive}'[Interactive mode]' \\
                                ':name'
                            ;;
                        list)
                            _arguments \\
                                '--format[Output format]:format:(text json yaml)' \\
                                '(-v --verbose)'{-v,--verbose}'[Show detailed information]'
                            ;;
                        show|apply|delete|export|validate)
                            _arguments \\
                                '--format[Output format]:format:(yaml json table)' \\
                                ':profile'
                        ;;
                        import)
                            _arguments \\
                                '--description[Profile description]' \\
                                ':file:_files' \\
                                ':name'
                            ;;
                    esac
                    ;;
                preset)
                    _describe 'preset command' preset_commands
                    case $words[3] in
                        list)
                            _arguments \\
                                '--category[Filter by category]:category:($preset_categories)' \\
                                '--format[Output format]:format:(text json yaml)' \\
                                '(-v --verbose)'{-v,--verbose}'[Show detailed information]'
                            ;;
                        show|apply|delete|validate)
                            _arguments \\
                                '--format[Output format]:format:(yaml json table)' \\
                                ':preset'
                        ;;
                        create)
                            _arguments \\
                                '--description[Preset description]' \\
                                '--category[Preset category]:category:(performance domain use_case system problem)' \\
                                ':name' \\
                                ':file:_files'
                        ;;
                        compare)
                            _arguments \\
                                '--format[Output format]:format:(table json yaml)' \\
                                ':preset1' \\
                                ':preset2'
                        ;;
                        search)
                            _arguments \\
                                '--category[Search in specific category]:category:($preset_categories)' \\
                                ':query'
                        ;;
                    esac
                    ;;
                env)
                    _describe 'env command' env_commands
                    case $words[3] in
                        set)
                            _arguments \\
                                '--export[Append to .env file]:file:_files' \\
                                '--permanent[Set permanently in shell config]' \\
                                ':variable' \\
                                ':value'
                        ;;
                        list)
                            _arguments \\
                                '--prefix[Filter by prefix]:prefix:' \\
                                '--format[Output format]:format:(text json export)'
                        ;;
                        export)
                            _arguments \\
                                '--output[Output file]:file:_files' \\
                                '--prefix[Filter by prefix]:prefix:' \\
                                '(-f --force)'{-f,--force}'[Overwrite existing file]'
                        ;;
                        load)
                            _arguments \\
                                '(--dry-run)--dry-run[Show what would be loaded]' \\
                                ':file:_files'
                        ;;
                        unset)
                            _arguments \\
                                '--remove-from[Remove from .env file]:file:_files' \\
                                ':variable'
                        ;;
                    esac
                    ;;
                validate)
                    _describe 'validate command' validate_commands
                    case $words[3] in
                        all)
                            _arguments \\
                                '--config[Config file]:file:_files' \\
                                '(-v --verbose)'{-v,--verbose}'[Verbose output]' \\
                                '--strict[Treat warnings as errors]'
                        ;;
                        config|quick)
                            _arguments \\
                                '(-v --verbose)'{-v,--verbose}'[Verbose output]' \\
                                '--format[Output format]:format:(text json yaml)' \\
                                ':file:_files'
                        ;;
                        profile)
                            _arguments \\
                                '(-v --verbose)'{-v,--verbose}'[Verbose output]' \\
                                ':profile'
                        ;;
                        preset)
                            _arguments \\
                                '(-v --verbose)'{-v,--verbose}'[Verbose output]' \\
                                ':preset'
                        ;;
                        env)
                            _arguments \\
                                '(-v --verbose)'{-v,--verbose}'[Verbose output]'
                        ;;
                        schema)
                            _arguments \\
                                '--format[Output format]:format:(json yaml)'
                        ;;
                    esac
                    ;;
            esac
            ;;
    esac
}

_evolve
"""


def get_bash_completion():
    """Get bash completion script"""
    return _BASH_COMPLETION


def get_zsh_completion():
    """Get zsh completion script"""
    return _ZSH_COMPLETION


def install_bash_completion():
    """Install bash completion to user's bashrc"""
    import os
    from pathlib import Path

    bashrc_paths = [
        Path.home() / '.bashrc',
        Path.home() / '.bash_profile',
    ]

    installed = False
    for bashrc in bashrc_paths:
        if bashrc.exists():
            with open(bashrc, 'a') as f:
                f.write('\n# OpenEvolve CLI completion\n')
                f.write(get_bash_completion())
            print(f"[OK] Installed bash completion to {bashrc}")
            installed = True
            break

    if not installed:
        print("Could not find .bashrc or .bash_profile")
        print("Add this to your shell config:")
        print(get_bash_completion())


def install_zsh_completion():
    """Install zsh completion to user's zshrc"""
    import os
    from pathlib import Path

    zshrc = Path.home() / '.zshrc'

    if zshrc.exists():
        with open(zshrc, 'a') as f:
            f.write('\n# OpenEvolve CLI completion\n')
            f.write(get_zsh_completion())
        print(f"[OK] Installed zsh completion to {zshrc}")
    else:
        print("Could not find .zshrc")
        print("Add this to your ~/.zshrc:")
        print(get_zsh_completion())


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--bash':
            print(get_bash_completion())
        elif sys.argv[1] == '--zsh':
            print(get_zsh_completion())
        elif sys.argv[1] == '--install-bash':
            install_bash_completion()
        elif sys.argv[1] == '--install-zsh':
            install_zsh_completion()
        else:
            print("Usage: python completion.py [--bash|--zsh|--install-bash|--install-zsh]")
            sys.exit(1)
