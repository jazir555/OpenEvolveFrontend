#!/bin/bash
# RESE Framework - Configuration Helper Scripts
# These scripts help with common configuration tasks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "ℹ $1"
}

# Validate configuration
validate_config() {
    local env_file="${1:-$SCRIPT_DIR/.env}"

    print_info "Validating configuration from: $env_file"

    if [ ! -f "$env_file" ]; then
        print_error "Configuration file not found: $env_file"
        print_info "Create one from .env.example: cp .env.example $env_file"
        exit 1
    fi

    cd "$SCRIPT_DIR"
    python -m config_validator --env-file "$env_file"

    if [ $? -eq 0 ]; then
        print_success "Configuration is valid!"
    else
        print_error "Configuration validation failed"
        exit 1
    fi
}

# Create .env from example
create_env() {
    local env_file="${1:-$SCRIPT_DIR/.env}"

    if [ -f "$env_file" ]; then
        print_warning "Configuration file already exists: $env_file"
        read -p "Overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Aborted"
            exit 0
        fi
    fi

    cp "$SCRIPT_DIR/.env.example" "$env_file"
    print_success "Created configuration file: $env_file"
    print_info "Edit it with your values before starting the adapter"
}

# Show configuration
show_config() {
    local env_file="${1:-$SCRIPT_DIR/.env}"

    print_info "Configuration from: $env_file"
    echo ""

    cd "$SCRIPT_DIR"
    python -m config_example
}

# Test configuration with dry run
dry_run() {
    local env_file="${1:-$SCRIPT_DIR/.env}"

    print_info "Testing configuration from: $env_file"
    echo ""

    cd "$SCRIPT_DIR"
    RESE_ENV="${RESE_ENV:-development}" python -m config_validator --env-file "$env_file" --verbose
}

# Set environment profile
set_profile() {
    local profile="$1"
    local env_file="${2:-$SCRIPT_DIR/.env}"

    if [ -z "$profile" ]; then
        print_error "Usage: $0 set-profile <development|staging|production> [env_file]"
        exit 1
    fi

    print_info "Setting profile: $profile"

    case "$profile" in
        development)
            print_info "Applying development profile..."
            # Update .env with development values
            sed -i.bak 's/^RESE_ENV=.*/RESE_ENV=development/' "$env_file"
            sed -i.bak 's/^RESE_LOG_LEVEL=.*/RESE_LOG_LEVEL=DEBUG/' "$env_file"
            sed -i.bak 's/^PHASE1_TIMEOUT_MS=.*/PHASE1_TIMEOUT_MS=10000/' "$env_file"
            sed -i.bak 's/^PHASE3_ITERATIONS=.*/PHASE3_ITERATIONS=5000/' "$env_file"
            sed -i.bak 's/^PHASE4_VALIDATION_LEVEL=.*/PHASE4_VALIDATION_LEVEL=1/' "$env_file"
            sed -i.bak 's/^PHASE1_ENABLE_LEAN4_INTEGRATION=.*/PHASE1_ENABLE_LEAN4_INTEGRATION=false/' "$env_file"
            sed -i.bak 's/^ENABLE_PROFILING=.*/ENABLE_PROFILING=true/' "$env_file"
            rm -f "$env_file.bak"
            ;;
        staging)
            print_info "Applying staging profile..."
            sed -i.bak 's/^RESE_ENV=.*/RESE_ENV=staging/' "$env_file"
            sed -i.bak 's/^RESE_LOG_LEVEL=.*/RESE_LOG_LEVEL=INFO/' "$env_file"
            sed -i.bak 's/^PHASE1_TIMEOUT_MS=.*/PHASE1_TIMEOUT_MS=30000/' "$env_file"
            sed -i.bak 's/^PHASE3_ITERATIONS=.*/PHASE3_ITERATIONS=50000/' "$env_file"
            sed -i.bak 's/^PHASE4_VALIDATION_LEVEL=.*/PHASE4_VALIDATION_LEVEL=3/' "$env_file"
            sed -i.bak 's/^PHASE1_ENABLE_LEAN4_INTEGRATION=.*/PHASE1_ENABLE_LEAN4_INTEGRATION=true/' "$env_file"
            sed -i.bak 's/^ENABLE_PROFILING=.*/ENABLE_PROFILING=true/' "$env_file"
            rm -f "$env_file.bak"
            ;;
        production)
            print_info "Applying production profile..."
            sed -i.bak 's/^RESE_ENV=.*/RESE_ENV=production/' "$env_file"
            sed -i.bak 's/^RESE_LOG_LEVEL=.*/RESE_LOG_LEVEL=WARN/' "$env_file"
            sed -i.bak 's/^PHASE1_TIMEOUT_MS=.*/PHASE1_TIMEOUT_MS=60000/' "$env_file"
            sed -i.bak 's/^PHASE3_ITERATIONS=.*/PHASE3_ITERATIONS=100000/' "$env_file"
            sed -i.bak 's/^PHASE4_VALIDATION_LEVEL=.*/PHASE4_VALIDATION_LEVEL=3/' "$env_file"
            sed -i.bak 's/^PHASE1_ENABLE_LEAN4_INTEGRATION=.*/PHASE1_ENABLE_LEAN4_INTEGRATION=true/' "$env_file"
            sed -i.bak 's/^ENABLE_PROFILING=.*/ENABLE_PROFILING=false/' "$env_file"
            rm -f "$env_file.bak"
            ;;
        *)
            print_error "Invalid profile: $profile"
            print_info "Valid profiles: development, staging, production"
            exit 1
            ;;
    esac

    print_success "Profile applied: $profile"
}

# Show help
show_help() {
    cat << EOF
RESE Framework - Configuration Helper Scripts

Usage: $0 <command> [arguments]

Commands:
    validate [env_file]     Validate configuration (default: .env)
    create [env_file]       Create .env from .env.example (default: .env)
    show [env_file]         Show current configuration (default: .env)
    dry-run [env_file]      Test configuration with dry run (default: .env)
    set-profile <profile>   Set environment profile
                           [env_file] (default: .env)

Examples:
    $0 validate
    $0 validate .env.production
    $0 create
    $0 show
    $0 dry-run
    $0 set-profile development
    $0 set-profile production .env.prod

Profiles:
    development     Fast iteration, detailed logging
    staging         Production-like with testing
    production      Maximum quality and resilience

EOF
}

# Main entry point
case "${1:-help}" in
    validate)
        validate_config "$2"
        ;;
    create)
        create_env "$2"
        ;;
    show)
        show_config "$2"
        ;;
    dry-run)
        dry_run "$2"
        ;;
    set-profile)
        set_profile "$2" "$3"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
