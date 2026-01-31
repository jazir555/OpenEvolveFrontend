# CLI Reference

**Version:** 1.0
**Date:** January 30, 2026
**Status:** Production Ready

Complete reference for all OpenEvolve CLI commands.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Commands](#2-core-commands)
3. [Config Commands](#3-config-commands)
4. [Profile Commands](#4-profile-commands)
5. [Preset Commands](#5-preset-commands)
6. [Environment Commands](#6-environment-commands)
7. [Validation Commands](#7-validation-commands)
8. [Info Commands](#8-info-commands)

---

## 1. Overview

### Installation

```bash
pip install openevolve-cli
```

### Basic Usage

```bash
evolve [OPTIONS] COMMAND [ARGS]
```

### Global Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--help` | `-h` | Show help message | - |
| `--version` | `-V` | Show version | - |
| `--verbose` | `-v` | Increase verbosity | INFO |
| `--quiet` | `-q` | Decrease verbosity | WARNING |
| `--config` | `-c` | Config file path | `./evolve.config.yaml` |
| `--profile` | `-p` | Profile name | - |
| `--output` | `-o` | Output format (json, yaml, table) | `table` |

---

## 2. Core Commands

### 2.1 evolve

**Description:** Run evolutionary optimization

**Syntax:**
```bash
evolve evolve [OPTIONS]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--problem` | Problem description | - |
| `--domain` | Problem domain | `general` |
| `--max-iterations` | Maximum iterations | `100` |
| `--max-evaluations` | Maximum evaluations | `100` |
| `--population-size` | Population size | `100` |
| `--evolution-mode` | Evolution mode | `auto` |
| `--objectives` | Optimization objectives (comma-separated) | - |
| `--enable-gauntlet` | Enable gauntlet | `true` |
| `--enable-knowledge-engine` | Enable knowledge engine | `false` |
| `--output-file` | Save results to file | - |

**Examples:**

```bash
# Basic evolution
evolve evolve --problem "Optimize portfolio allocation" --domain finance

# With custom parameters
evolve evolve \
  --problem "Develop trading strategy" \
  --domain trading \
  --max-evaluations 50 \
  --evolution-mode pes

# Save results
evolve evolve \
  --problem "Optimize chemical reaction" \
  --domain science \
  --output-file results.json

# With objectives
evolve evolve \
  --problem "Portfolio optimization" \
  --domain finance \
  --objectives return,risk,sharpe_ratio
```

**Exit Codes:**
- `0` - Success
- `1` - General error
- `2` - Validation error
- `3` - Execution error
- `4` - Timeout

---

### 2.2 quick-evolve

**Description:** Quick evolution with time limit

**Syntax:**
```bash
evolve quick-evolve [OPTIONS]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--problem` | Problem description | - |
| `--domain` | Problem domain | `general` |
| `--max-minutes` | Maximum execution time (minutes) | `5` |
| `--output-file` | Save results to file | - |

**Examples:**

```bash
# 5-minute evolution
evolve quick-evolve \
  --problem "Optimize landing page" \
  --domain web_design \
  --max-minutes 5

# 10-minute evolution
evolve quick-evolve \
  --problem "Optimize portfolio" \
  --domain finance \
  --max-minutes 10
```

---

### 2.3 batch-evolve

**Description:** Run multiple evolutions in parallel

**Syntax:**
```bash
evolve batch-evolve [OPTIONS]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--problems-file` | File with problems (one per line) | - |
| `--domain` | Problem domain | `general` |
| `--max-parallel` | Maximum parallel jobs | `4` |
| `--output-dir` | Output directory | `./results` |
| `--continue-on-error` | Continue on individual errors | `false` |

**Examples:**

```bash
# From file
echo "Optimize tech portfolio
Optimize healthcare portfolio
Optimize energy portfolio" > problems.txt

evolve batch-evolve \
  --problems-file problems.txt \
  --domain finance \
  --max-parallel 3 \
  --output-dir results

# Continue on error
evolve batch-evolve \
  --problems-file problems.txt \
  --continue-on-error
```

---

## 3. Config Commands

### 3.1 config validate

**Description:** Validate configuration file

**Syntax:**
```bash
evolve config validate [OPTIONS] [CONFIG_FILE]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--strict` | Enable strict validation | `false` |
| `--show-warnings` | Show warnings | `true` |

**Examples:**

```bash
# Validate config
evolve config validate evolve.config.yaml

# Strict validation
evolve config validate --strict evolve.config.yaml

# No warnings
evolve config validate --show-warnings false evolve.config.yaml
```

**Exit Codes:**
- `0` - Valid
- `1` - Invalid

---

### 3.2 config show

**Description:** Show current configuration

**Syntax:**
```bash
evolve config show [OPTIONS]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--effective` | Show effective config (after merging) | `false` |
| `--source` | Show config source | `false` |
| `--format` | Output format (json, yaml, table) | `yaml` |

**Examples:**

```bash
# Show config
evolve config show

# Effective config
evolve config show --effective

# With sources
evolve config show --source

# JSON format
evolve config show --format json
```

---

### 3.3 config defaults

**Description:** Show default configuration

**Syntax:**
```bash
evolve config defaults [OPTIONS]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--format` | Output format (json, yaml) | `yaml` |

**Examples:**

```bash
# Show defaults
evolve config defaults

# JSON format
evolve config defaults --format json
```

---

### 3.4 config merge

**Description:** Merge multiple configuration files

**Syntax:**
```bash
evolve config merge [OPTIONS] BASE OVERRIDE...
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--output` | `-o` | Output file | - |
| `--format` | Output format (yaml, json) | `yaml` |

**Examples:**

```bash
# Merge two files
evolve config merge base.yaml override.yaml -o merged.yaml

# Merge multiple files
evolve config merge base.yaml override1.yaml override2.yaml -o final.yaml

# JSON output
evolve config merge base.yaml override.yaml --format json
```

---

### 3.5 config diff

**Description:** Compare two configuration files

**Syntax:**
```bash
evolve config diff [OPTIONS] CONFIG1 CONFIG2
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--unified` | Unified diff format | `false` |
| `--context` | Context lines | `3` |

**Examples:**

```bash
# Compare configs
evolve config config diff base.yaml updated.yaml

# Unified format
evolve config diff --unified base.yaml updated.yaml

# More context
evolve config diff --context 5 base.yaml updated.yaml
```

---

### 3.6 config init

**Description:** Initialize new configuration file

**Syntax:**
```bash
evolve config init [OPTIONS] [OUTPUT_FILE]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--preset` | Start from preset | `balanced` |
| `--profile` | Start from profile | - |
| `--interactive` | Interactive mode | `false` |

**Examples:**

```bash
# Initialize with defaults
evolve config init evolve.config.yaml

# From preset
evolve config init --preset finance evolve.config.yaml

# From profile
evolve config init --profile prod evolve.config.yaml

# Interactive mode
evolve config init --interactive evolve.config.yaml
```

---

## 4. Profile Commands

### 4.1 profile list

**Description:** List available profiles

**Syntax:**
```bash
evolve profile list [OPTIONS]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--verbose` | Show profile details | `false` |
| `--format` | Output format (table, json) | `table` |

**Examples:**

```bash
# List profiles
evolve profile list

# With details
evolve profile list --verbose

# JSON format
evolve profile list --format json
```

---

### 4.2 profile show

**Description:** Show profile details

**Syntax:**
```bash
evolve profile show [OPTIONS] PROFILE_NAME
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--format` | Output format (yaml, json) | `yaml` |
| `--resolved` | Show resolved config (with inheritance) | `false` |

**Examples:**

```bash
# Show profile
evolve profile show prod

# Resolved config
evolve profile show prod --resolved

# JSON format
evolve profile show prod --format json
```

---

### 4.3 profile create

**Description:** Create new profile

**Syntax:**
```bash
evolve profile create [OPTIONS] PROFILE_NAME
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--inherit` | Inherit from profile | - |
| `--from-preset` | Start from preset | - |
| `--interactive` | Interactive mode | `false` |
| `--description` | Profile description | - |

**Examples:**

```bash
# Create profile
evolve profile create my_profile \
  --inherit prod \
  --description "My custom production profile"

# From preset
evolve profile create my_finance \
  --from-preset finance

# Interactive
evolve profile create my_profile --interactive
```

---

### 4.4 profile validate

**Description:** Validate profile

**Syntax:**
```bash
evolve profile validate [OPTIONS] PROFILE_NAME
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--strict` | Enable strict validation | `false` |

**Examples:**

```bash
# Validate profile
evolve profile validate prod

# Strict validation
evolve profile validate prod --strict
```

---

### 4.5 profile delete

**Description:** Delete custom profile

**Syntax:**
```bash
evolve profile delete [OPTIONS] PROFILE_NAME
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--force` | Skip confirmation | `false` |

**Examples:**

```bash
# Delete profile
evolve profile delete my_profile

# Force delete
evolve profile delete my_profile --force
```

---

## 5. Preset Commands

### 5.1 preset list

**Description:** List available presets

**Syntax:**
```bash
evolve preset list [OPTIONS]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--category` | Filter by category | - |
| `--verbose` | Show preset details | `false` |

**Examples:**

```bash
# List all presets
evolve preset list

# Filter by category
evolve preset list --category performance

# With details
evolve preset list --verbose
```

---

### 5.2 preset show

**Description:** Show preset details

**Syntax:**
```bash
evolve preset show [OPTIONS] PRESET_NAME
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--format` | Output format (yaml, json) | `yaml` |

**Examples:**

```bash
# Show preset
evolve preset show finance

# JSON format
evolve preset show finance --format json
```

---

### 5.3 preset apply

**Description:** Apply preset to config file

**Syntax:**
```bash
evolve preset apply [OPTIONS] PRESET_NAME
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--output` | `-o` | Output file | `evolve.config.yaml` |
| `--merge` | Merge with existing config | `false` |

**Examples:**

```bash
# Apply preset
evolve preset apply finance -o evolve.config.yaml

# Merge with existing
evolve preset apply finance --merge -o evolve.config.yaml
```

---

### 5.4 preset create

**Description:** Create custom preset

**Syntax:**
```bash
evolve preset create [OPTIONS] PRESET_NAME
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--base` | Base preset | `balanced` |
| `--interactive` | Interactive mode | `false` |
| `--description` | Preset description | - |
| `--category` | Preset category | `custom` |

**Examples:**

```bash
# Create preset
evolve preset create my_custom \
  --base balanced \
  --description "My custom preset" \
  --category performance

# Interactive
evolve preset create my_preset --interactive
```

---

## 6. Environment Commands

### 6.1 env show

**Description:** Show environment variables

**Syntax:**
```bash
evolve env show [OPTIONS]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--prefix` | Filter by prefix | `EVOLVE_` |
| `--format` | Output format (env, json) | `env` |

**Examples:**

```bash
# Show all env vars
evolve env show

# JSON format
evolve env show --format json

# Custom prefix
evolve env show --prefix MYAPP_
```

---

### 6.2 env export

**Description:** Export environment to file

**Syntax:**
```bash
evolve env export [OPTIONS] [OUTPUT_FILE]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--format` | Output format (env, json) | `env` |
| `--prefix` | Filter by prefix | `EVOLVE_` |

**Examples:**

```bash
# Export to .env
evolve env export > .env

# Export to file
evolve env export -o .env

# JSON format
evolve env export --format json > env.json
```

---

### 6.3 env load

**Description:** Load environment from file

**Syntax:**
```bash
evolve env load [OPTIONS] ENV_FILE
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--export` | Export to shell environment | `false` |
| `--override` | Override existing variables | `false` |

**Examples:**

```bash
# Load from file
evolve env load .env

# Export to shell
eval $(evolve env load --export .env)

# Override existing
evolve env load .env --override
```

---

### 6.4 env validate

**Description:** Validate environment variables

**Syntax:**
```bash
evolve env validate [OPTIONS]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--required` | Required variables (comma-separated) | - |
| `--strict` | Enable strict validation | `false` |

**Examples:**

```bash
# Validate env
evolve env validate

# With required vars
evolve env validate --required API_KEY,API_SECRET

# Strict validation
evolve env validate --strict
```

---

## 7. Validation Commands

### 7.1 validate problem

**Description:** Validate problem description

**Syntax:**
```bash
evolve validate problem [OPTIONS] PROBLEM
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--domain` | Check domain-specific rules | - |
| `--format` | Output format (text, json) | `text` |

**Examples:**

```bash
# Validate problem
evolve validate problem "Optimize portfolio allocation"

# With domain
evolve validate problem "Optimize portfolio" --domain finance

# JSON output
evolve validate problem "..." --format json
```

---

### 7.2 validate config

**Description:** Validate configuration (same as config validate)

**Syntax:**
```bash
evolve validate config [OPTIONS] [CONFIG_FILE]
```

---

### 7.3 validate constraints

**Description:** Validate constraints

**Syntax:**
```bash
evolve validate constraints [OPTIONS] CONSTRAINTS_FILE
```

**Examples:**

```bash
# Validate constraints
evolve validate validate constraints constraints.yaml
```

---

## 8. Info Commands

### 8.1 info version

**Description:** Show version information

**Syntax:**
```bash
evolve info version [OPTIONS]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--detailed` | Show detailed version info | `false` |

**Examples:**

```bash
# Show version
evolve info version

# Detailed info
evolve info version --detailed
```

---

### 8.2 info system

**Description:** Show system information

**Syntax:**
```bash
evolve info system [OPTIONS]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--format` | Output format (text, json) | `text` |

**Examples:**

```bash
# System info
evolve info system

# JSON format
evolve info system --format json
```

---

### 8.3 info paths

**Description:** Show system paths

**Syntax:**
```bash
evolve info paths [OPTIONS]
```

**Examples:**

```bash
# Show paths
evolve info paths
```

---

## Command Completion

### Bash

```bash
# Enable completion
evolve completion bash > /etc/bash_completion.d/evolve
source /etc/bash_completion.d/evolve
```

### Zsh

```bash
# Enable completion
evolve completion zsh > ~/.zsh/completion/_evolve
echo "fpath=(~/.zsh/completion \$fpath)" >> ~/.zshrc
echo "autoload -U compinit && compinit" >> ~/.zshrc
```

### Fish

```bash
# Enable completion
evolve completion fish > ~/.config/fish/completions/evolve.fish
```

---

**End of CLI Reference**

For more information:
- [Configuration Guide](CONFIGURATION_GUIDE.md) - Master configuration guide
- [Configuration Examples](CONFIGURATION_EXAMPLES.md) - Working examples
