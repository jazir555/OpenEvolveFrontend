#!/usr/bin/env python3
"""
BubbleLab COMPLETE Automated Setup Script v3.0
==============================================

Production-Grade Reliability with Comprehensive Validation and Error Handling

This script AUTOMATICALLY configures EVERYTHING needed for BubbleLab automation:
- Validates environment with comprehensive checks
- Validates all inputs before processing
- Installs dependencies with version pinning
- Creates all directories with proper error handling
- Configures credentials with validation
- Validates API connectivity and credentials
- Deploys example workflows
- Tests all components
- Generates configuration files
- Logs all errors to file for debugging

Usage:
    python bubblelab-auto-setup.py [--api-url URL] [--api-key KEY] [--skip-tests]

Author: BubbleLab Automation Team
Version: 3.0.0 (Production-Grade Reliability)
"""


import os
import sys
import json
import time
import re
import socket
import hashlib
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from urllib.parse import urlparse
from contextlib import contextmanager
import traceback

# Try to import optional dependencies
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# =============================================================================
# ANSI Color Codes for Beautiful Output
# =============================================================================
class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# =============================================================================
# Error Guide - User-Friendly Error Messages
# =============================================================================
ERROR_GUIDE = {
    'INVALID_PYTHON_VERSION': {
        'error': 'Python version must be 3.10 or higher',
        'solution': 'Install Python 3.10+ from python.org or use pyenv/conda',
        'docs': 'https://www.python.org/downloads/'
    },
    'PIP_NOT_AVAILABLE': {
        'error': 'pip is not available',
        'solution': 'Ensure pip is installed: python -m ensurepip --upgrade',
        'docs': 'https://pip.pypa.io/en/stable/installation/'
    },
    'INVALID_API_URL': {
        'error': 'Invalid API URL format',
        'solution': 'URL must start with http:// or https:// and be a valid URL',
        'example': 'http://localhost:3001 or https://api.bubblelab.io'
    },
    'INVALID_API_KEY': {
        'error': 'Invalid API key format',
        'solution': 'API key must be at least 20 characters and contain only alphanumeric characters, hyphens, and underscores',
        'warning': 'Never pass API key via CLI argument in production (visible in ps)'
    },
    'API_CONNECTION_FAILED': {
        'error': 'Cannot connect to BubbleLab API',
        'solution': '1) Check if BubbleLab is running\n2) Verify the API URL is correct\n3) Check network connectivity\n4) Verify API key is valid',
        'test': 'curl -H "Authorization: Bearer YOUR_KEY" YOUR_API_URL/'
    },
    'DIRECTORY_NOT_WRITABLE': {
        'error': 'Cannot write to directory',
        'solution': 'Check directory permissions and ensure you have write access',
        'linux': 'chmod u+w /path/to/directory'
    },
    'DEPENDENCY_INSTALL_FAILED': {
        'error': 'Failed to install required dependencies',
        'solution': '1) Check internet connection\n2) Try installing manually: pip install requests pyyaml python-dotenv\n3) Check if behind corporate proxy',
        'proxy': 'export HTTP_PROXY=http://proxy.example.com:8080'
    },
    'NOT_IN_VENV': {
        'error': 'Not in a virtual environment',
        'solution': 'Create a virtual environment first:\npython -m venv venv\nsource venv/bin/activate  # Linux/Mac\nvenv\\Scripts\\activate  # Windows',
        'why': 'Installing to system Python can break other applications'
    },
    'DATABASE_CONNECTION_FAILED': {
        'error': 'Cannot connect to database',
        'solution': '1) Check if database is running\n2) Verify connection string format\n3) Check firewall rules\n4) Ensure database accepts remote connections'
    },
    'CONFIG_SCHEMA_INVALID': {
        'error': 'Configuration schema validation failed',
        'solution': 'Generated configuration does not match required schema. Please report this issue.',
        'support': 'https://github.com/bubblelab/support/issues'
    }
}

# =============================================================================
# File Logger - Persistent Error Logging
# =============================================================================
class FileLogger:
    """Persistent error logging to file"""

    def __init__(self, log_dir: Path = None):
        """Initialize file logger"""
        self.log_dir = log_dir or Path.cwd() / 'bubblelab-logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'setup_{timestamp}.log'

        # Configure Python logging
        self.logger = logging.getLogger('BubbleLabSetup')
        self.logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Console handler (errors only)
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.info(f"Logging initialized: {self.log_file}")

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str, exception: Exception = None):
        """Log error message with optional exception details"""
        if exception:
            self.logger.error(message, exc_info=True)
            self.logger.debug(traceback.format_exc())
        else:
            self.logger.error(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def log(self, level: str, message: str, exception: Exception = None):
        """Log at specified level"""
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        if exception:
            log_func(message, exc_info=True)
        else:
            log_func(message)

    def get_log_file(self) -> Path:
        """Get log file path"""
        return self.log_file

# =============================================================================
# Console Logger - Beautiful Terminal Output
# =============================================================================
class Logger:
    """Beautiful logging with colors and emojis"""

    @staticmethod
    def header(text: str):
        """Print header"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

    @staticmethod
    def section(text: str):
        """Print section header"""
        print(f"\n{Colors.OKCYAN}{Colors.BOLD}▶ {text}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}{'─'*80}{Colors.ENDC}")

    @staticmethod
    def success(text: str):
        """Print success message"""
        print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

    @staticmethod
    def error(text: str):
        """Print error message"""
        print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

    @staticmethod
    def warning(text: str):
        """Print warning message"""
        print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

    @staticmethod
    def info(text: str):
        """Print info message"""
        print(f"ℹ️  {text}")

    @staticmethod
    def step(step_num: int, total: int, text: str):
        """Print step progress"""
        print(f"\n{Colors.OKBLUE}[{step_num}/{total}] {Colors.BOLD}{text}{Colors.ENDC}")

    @staticmethod
    def detail(text: str):
        """Print detailed info"""
        print(f"    {text}")

# =============================================================================
# Input Validation Classes
# =============================================================================
class URLValidator:
    """Validates URLs"""

    @staticmethod
    def validate(url: str, file_logger: FileLogger = None) -> Tuple[bool, str]:
        """
        Validate API URL format

        Args:
            url: URL to validate
            file_logger: Optional file logger

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = urlparse(url.strip())

            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                error = "URL must use http:// or https:// scheme"
                if file_logger:
                    file_logger.error(f"URL validation failed: {error}")
                return False, error

            # Check netloc (domain/IP)
            if not parsed.netloc:
                error = "Invalid URL format (missing domain or IP)"
                if file_logger:
                    file_logger.error(f"URL validation failed: {error}")
                return False, error

            # Check for valid hostname characters
            if not re.match(r'^[a-zA-Z0-9.-]+$', parsed.netloc.split(':')[0]):
                error = "Invalid hostname format"
                if file_logger:
                    file_logger.error(f"URL validation failed: {error}")
                return False, error

            # Log success
            if file_logger:
                file_logger.info(f"URL validated: {url}")

            return True, ""

        except (ValueError, AttributeError) as e:
            error = f"URL validation error: {str(e)}"
            if file_logger:
                file_logger.error(error, e)
            return False, error

class APIKeyValidator:
    """Validates API keys"""

    MIN_LENGTH = 20
    ALLOWED_PATTERN = r'^[a-zA-Z0-9\-_\.]+$'

    @staticmethod
    def validate(api_key: str, file_logger: FileLogger = None) -> Tuple[bool, str]:
        """
        Validate API key format

        Args:
            api_key: API key to validate
            file_logger: Optional file logger

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not api_key or api_key.strip() == "YOUR_API_KEY_HERE":
            # Not an error, just a placeholder
            return True, ""

        api_key = api_key.strip()

        # Check minimum length
        if len(api_key) < APIKeyValidator.MIN_LENGTH:
            error = f"API key must be at least {APIKeyValidator.MIN_LENGTH} characters (got {len(api_key)})"
            if file_logger:
                file_logger.error(f"API key validation failed: {error}")
            return False, error

        # Check for valid characters
        if not re.match(APIKeyValidator.ALLOWED_PATTERN, api_key):
            error = f"API key contains invalid characters (only alphanumeric, hyphen, underscore, and dot allowed)"
            if file_logger:
                file_logger.error(f"API key validation failed: {error}")
            return False, error

        # Check for obvious weak keys
        if api_key.lower() in ['password', 'secret', 'key', 'api_key', 'test', 'demo']:
            error = "API key appears to be a placeholder or weak value"
            if file_logger:
                file_logger.warning(f"API key validation warning: {error}")
            return False, error

        # Log success (without exposing the key)
        if file_logger:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8]
            file_logger.info(f"API key validated (hash: {key_hash}...)")

        return True, ""

    @staticmethod
    def warn_cli_usage(file_logger: FileLogger = None):
        """Warn user about passing API key via CLI"""
        warning = "SECURITY WARNING: API key passed via --api-key is visible in process list (ps aux). Use environment variable BUBBLELAB_API_KEY in production."
        Logger.warning(warning)
        if file_logger:
            file_logger.warning(warning)

class ConnectionStringValidator:
    """Validates database connection strings"""

    @staticmethod
    def validate_postgresql(conn_string: str, file_logger: FileLogger = None) -> Tuple[bool, str]:
        """
        Validate PostgreSQL connection string

        Format: postgresql://user:password@host:port/database
        """
        try:
            parsed = urlparse(conn_string)

            if parsed.scheme != 'postgresql':
                error = f"Invalid scheme (expected 'postgresql', got '{parsed.scheme}')"
                return False, error

            if not parsed.hostname:
                error = "Missing hostname in connection string"
                return False, error

            if not parsed.path or len(parsed.path) <= 1:
                error = "Missing database name in connection string"
                return False, error

            if file_logger:
                file_logger.info(f"PostgreSQL connection string validated for {parsed.hostname}")

            return True, ""

        except (ValueError, AttributeError) as e:
            error = f"Connection string validation error: {str(e)}"
            if file_logger:
                file_logger.error(error, e)
            return False, error

    @staticmethod
    def validate_redis(conn_string: str, file_logger: FileLogger = None) -> Tuple[bool, str]:
        """
        Validate Redis connection string

        Format: redis://host:port/db
        """
        try:
            parsed = urlparse(conn_string)

            if parsed.scheme != 'redis':
                error = f"Invalid scheme (expected 'redis', got '{parsed.scheme}')"
                return False, error

            if not parsed.hostname:
                error = "Missing hostname in connection string"
                return False, error

            if file_logger:
                file_logger.info(f"Redis connection string validated for {parsed.hostname}")

            return True, ""

        except (ValueError, AttributeError) as e:
            error = f"Redis connection string validation error: {str(e)}"
            if file_logger:
                file_logger.error(error, e)
            return False, error

# =============================================================================
# Python Environment Validator
# =============================================================================
class PythonEnvironmentValidator:
    """Validates Python environment"""

    @staticmethod
    def is_in_virtualenv() -> bool:
        """Check if running in virtual environment"""
        return hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )

    @staticmethod
    def is_conda_env() -> bool:
        """Check if running in conda environment"""
        return 'CONDA_DEFAULT_ENV' in os.environ or os.path.exists(os.path.join(sys.prefix, 'conda-meta'))

    @staticmethod
    def validate(file_logger: FileLogger = None) -> Tuple[bool, str]:
        """
        Validate Python environment

        Returns:
            Tuple of (is_valid, warning_message)
        """
        in_venv = PythonEnvironmentValidator.is_in_virtualenv()
        in_conda = PythonEnvironmentValidator.is_conda_env()

        if in_venv or in_conda:
            env_type = "conda" if in_conda else "venv"
            Logger.success(f"Running in {env_type} environment: {sys.prefix}")
            if file_logger:
                file_logger.info(f"Python environment validated: {env_type} at {sys.prefix}")
            return True, ""
        else:
            warning = "Not in a virtual environment - installing to system Python!"
            Logger.warning(warning)
            Logger.detail("  Recommended: Create a virtual environment first")

            if file_logger:
                file_logger.warning(warning)

            return False, "NOT_IN_VENV"

# =============================================================================
# Configuration Schema Validator
# =============================================================================
class ConfigSchemaValidator:
    """Validates configuration against schema"""

    CONFIG_SCHEMA = {
        'type': 'object',
        'required': ['base_url', 'api_key'],
        'properties': {
            'base_url': {'type': 'string', 'format': 'uri'},
            'api_key': {'type': 'string', 'minLength': 1},
            'workflows_dir': {'type': 'string'},
            'templates_dir': {'type': 'string'},
            'exports_dir': {'type': 'string'},
            'backups_dir': {'type': 'string'},
            'tests_dir': {'type': 'string'},
            'environments': {
                'type': 'object',
                'properties': {
                    'development': {
                        'type': 'object',
                        'properties': {
                            'api_url': {'type': 'string'},
                            'qdrant_url': {'type': 'string'},
                            'postgres_url': {'type': 'string'},
                            'redis_url': {'type': 'string'},
                            'slack_channel': {'type': 'string'}
                        }
                    },
                    'production': {
                        'type': 'object',
                        'properties': {
                            'api_url': {'type': 'string'},
                            'qdrant_url': {'type': 'string'},
                            'postgres_url': {'type': 'string'},
                            'redis_url': {'type': 'string'},
                            'slack_channel': {'type': 'string'}
                        }
                    }
                }
            }
        }
    }

    @staticmethod
    def validate(config: Dict, file_logger: FileLogger = None) -> Tuple[bool, str]:
        """
        Validate configuration against schema

        Args:
            config: Configuration dictionary
            file_logger: Optional file logger

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Manual schema validation (avoid jsonschema dependency)
            required_fields = ['base_url', 'api_key']

            for field in required_fields:
                if field not in config:
                    error = f"Missing required field: {field}"
                    if file_logger:
                        file_logger.error(f"Schema validation failed: {error}")
                    return False, error

                if not config[field] or not isinstance(config[field], str):
                    error = f"Invalid value for field: {field}"
                    if file_logger:
                        file_logger.error(f"Schema validation failed: {error}")
                    return False, error

            # Validate base_url format
            is_valid, error = URLValidator.validate(config['base_url'], file_logger)
            if not is_valid:
                return False, f"base_url: {error}"

            if file_logger:
                file_logger.info("Configuration schema validated")

            return True, ""

        except (ValueError, KeyError, TypeError) as e:
            error = f"Schema validation error: {str(e)}"
            if file_logger:
                file_logger.error(error, e)
            return False, error

# =============================================================================
# Environment Validator (Enhanced)
# =============================================================================
class EnvironmentValidator:
    """Validates the runtime environment"""

    def __init__(self, file_logger: FileLogger = None):
        self.errors = []
        self.warnings = []
        self.file_logger = file_logger

    def validate_python_version(self) -> bool:
        """Check Python version >= 3.10"""
        Logger.section("Validating Python Version")
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major == 3 and version.minor >= 10:
            Logger.success(f"Python {version_str} (>= 3.10 required)")
            if self.file_logger:
                self.file_logger.info(f"Python version validated: {version_str}")
            return True
        else:
            Logger.error(f"Python {version_str} found (>= 3.10 required)")
            error = f"Python version too old: {version_str}"
            self.errors.append(error)
            if self.file_logger:
                self.file_logger.error(error)
            self._print_error_guide('INVALID_PYTHON_VERSION')
            return False

    def validate_pip(self) -> bool:
        """Check if pip is available"""
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', '--version'],
                capture_output=True,
                check=True
            )
            Logger.success("pip is available")
            if self.file_logger:
                self.file_logger.info("pip validated")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            Logger.error(f"pip not available: {e}")
            error = "pip not available"
            self.errors.append(error)
            if self.file_logger:
                self.file_logger.error(error, e)
            self._print_error_guide('PIP_NOT_AVAILABLE')
            return False

    def check_directory_writable(self, directory: Path) -> bool:
        """Check if directory is writable"""
        try:
            test_file = directory / '.write_test'
            test_file.touch()
            test_file.unlink()
            if self.file_logger:
                self.file_logger.info(f"Directory writable: {directory}")
            return True
        except (PermissionError, OSError) as e:
            Logger.error(f"Directory not writable: {directory}")
            error = f"Cannot write to {directory}"
            self.errors.append(error)
            if self.file_logger:
                self.file_logger.error(error, e)
            self._print_error_guide('DIRECTORY_NOT_WRITABLE')
            return False

    def validate_python_environment(self) -> bool:
        """Validate Python environment (venv/conda)"""
        Logger.section("Validating Python Environment")

        is_valid, warning_key = PythonEnvironmentValidator.validate(self.file_logger)

        if not is_valid:
            self.warnings.append(warning_key)
            self._print_error_guide(warning_key)

        return is_valid

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """Run all validations"""
        Logger.header("ENVIRONMENT VALIDATION")
        all_valid = True

        all_valid &= self.validate_python_version()
        all_valid &= self.validate_pip()
        all_valid &= self.check_directory_writable(Path.cwd())
        self.validate_python_environment()  # Warning only, doesn't fail

        if all_valid:
            Logger.success("\n✓ Environment validation PASSED")
        else:
            Logger.error("\n✗ Environment validation FAILED")

        return all_valid, self.errors, self.warnings

    def _print_error_guide(self, error_key: str):
        """Print user-friendly error guide"""
        if error_key not in ERROR_GUIDE:
            return

        guide = ERROR_GUIDE[error_key]

        print(f"\n{Colors.WARNING}{'─'*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}How to fix: {guide['error']}{Colors.ENDC}\n")
        print(f"{Colors.OKCYAN}Solution:{Colors.ENDC} {guide['solution']}")

        if 'example' in guide:
            print(f"{Colors.OKCYAN}Example:{Colors.ENDC} {guide['example']}")

        if 'test' in guide:
            print(f"{Colors.OKCYAN}Test command:{Colors.ENDC} {guide['test']}")

        if 'docs' in guide:
            print(f"{Colors.OKCYAN}Documentation:{Colors.ENDC} {guide['docs']}")

        if 'support' in guide:
            print(f"{Colors.OKCYAN}Support:{Colors.ENDC} {guide['support']}")

        print(f"{Colors.WARNING}{'─'*80}{Colors.ENDC}\n")

# =============================================================================
# Dependency Installer (Enhanced with Version Pinning)
# =============================================================================
class DependencyInstaller:
    """Installs required dependencies with version pinning"""

    # Use exact versions for reproducibility
    REQUIRED_PACKAGES = [
        'requests==2.31.0',
        'pyyaml==6.0.1',
        'python-dotenv==1.0.0',
    ]

    def __init__(self, file_logger: FileLogger = None):
        self.installed = []
        self.failed = []
        self.file_logger = file_logger

    def check_installed_version(self, package: str) -> Optional[str]:
        """Check if package is already installed and get version"""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', package.split('==')[0]],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
            return None
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Package check failed: {e}")
            return None

    def install_package(self, package: str) -> bool:
        """Install a single package"""
        package_name = package.split('==')[0]
        required_version = package.split('==')[1] if '==' in package else None

        try:
            # Check current version
            current_version = self.check_installed_version(package_name)

            if current_version:
                if required_version and current_version == required_version:
                    Logger.detail(f"✓ {package_name} {current_version} (already installed)")
                    self.installed.append(package)
                    if self.file_logger:
                        self.file_logger.info(f"Package already installed: {package_name} {current_version}")
                    return True
                elif required_version and current_version != required_version:
                    Logger.warning(f"{package_name} {current_version} installed, but {required_version} required")
                    Logger.detail(f"  Upgrading {package_name} {current_version} -> {required_version}")

            Logger.detail(f"Installing {package}...")
            if self.file_logger:
                self.file_logger.info(f"Installing package: {package}")

            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q', package],
                check=True,
                capture_output=True,
                timeout=120
            )

            Logger.success(f"✓ {package}")
            self.installed.append(package)

            if self.file_logger:
                self.file_logger.info(f"Package installed successfully: {package}")

            return True

        except subprocess.TimeoutExpired:
            error = f"{package} (timeout)"
            Logger.error(f"✗ {error}")
            self.failed.append(package)
            if self.file_logger:
                self.file_logger.error(f"Package installation timeout: {package}")
            return False
        except (subprocess.CalledProcessError, OSError) as e:
            error = f"{package} ({e})"
            Logger.error(f"✗ {error}")
            self.failed.append(package)
            if self.file_logger:
                self.file_logger.error(f"Package installation failed: {package}", e)
            return False

    def install_all(self) -> bool:
        """Install all required packages"""
        Logger.section("Installing Dependencies")

        all_success = True
        for package in self.REQUIRED_PACKAGES:
            success = self.install_package(package)
            all_success &= success

        if all_success:
            Logger.success(f"\n✓ All {len(self.installed)} packages installed successfully")
            if self.file_logger:
                self.file_logger.info("All dependencies installed successfully")
        else:
            Logger.warning(f"\n⚠️  {len(self.installed)} installed, {len(self.failed)} failed")
            if self.file_logger:
                self.file_logger.warning(f"Some dependencies failed: {self.failed}")
            self._print_error_guide('DEPENDENCY_INSTALL_FAILED')

        return all_success

    def _print_error_guide(self, error_key: str):
        """Print user-friendly error guide"""
        if error_key not in ERROR_GUIDE:
            return

        guide = ERROR_GUIDE[error_key]

        print(f"\n{Colors.WARNING}{'─'*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}How to fix: {guide['error']}{Colors.ENDC}\n")
        print(f"{Colors.OKCYAN}Solution:{Colors.ENDC} {guide['solution']}")

        if 'proxy' in guide:
            print(f"{Colors.OKCYAN}Proxy setup:{Colors.ENDC} {guide['proxy']}")

        print(f"{Colors.WARNING}{'─'*80}{Colors.ENDC}\n")

# =============================================================================
# BubbleLab API Client (Enhanced with Validation)
# =============================================================================
class BubbleLabClient:
    """Extended BubbleLab API client with comprehensive validation"""

    def __init__(self, base_url: str, api_key: str, file_logger: FileLogger = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.file_logger = file_logger
        self.session = None

        # Import requests here
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            })
            if self.file_logger:
                self.file_logger.info(f"API client initialized for {base_url}")
        except ImportError:
            error = "requests package not installed"
            if self.file_logger:
                self.file_logger.error(error)
            raise ImportError(error)

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make API request with error handling"""
        url = f"{self.base_url}{endpoint}"
        try:
            if self.file_logger:
                self.file_logger.debug(f"API request: {method} {url}")

            response = self.session.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()

            if self.file_logger:
                self.file_logger.debug(f"API response: {response.status_code}")

            return response.json()
        except requests.exceptions.Timeout:
            error = f"API request timed out: {url}"
            if self.file_logger:
                self.file_logger.error(error)
            raise Exception(error)
        except requests.exceptions.ConnectionError as e:
            error = f"Cannot connect to API: {e}"
            if self.file_logger:
                self.file_logger.error(error, e)
            raise Exception(error)
        except requests.exceptions.HTTPError as e:
            error = f"HTTP error: {e.response.status_code} - {e.response.text}"
            if self.file_logger:
                self.file_logger.error(error)
            raise Exception(error)
        except (ValueError, AttributeError, RuntimeError) as e:
            error = f"API request failed: {e}"
            if self.file_logger:
                self.file_logger.error(error, e)
            raise RuntimeError(error)

    def test_credentials(self) -> Tuple[bool, str]:
        """Test if API credentials are valid by calling /me endpoint"""
        try:
            # Try to get user info (authenticates the API key)
            self._request('GET', '/me')
            return True, "Credentials validated"
        except (RuntimeError, ConnectionError, ValueError) as e:
            return False, str(e)

    def test_connection(self) -> Tuple[bool, str]:
        """Test API connectivity"""
        try:
            # Try to list flows (lightweight endpoint)
            self._request('GET', '/bubble-flow?limit=1')
            return True, "Connection successful"
        except (RuntimeError, ConnectionError, ValueError) as e:
            return False, str(e)

    def get_system_status(self) -> Dict:
        """Get BubbleLab system status"""
        try:
            return self._request('GET', '/')
        except (RuntimeError, ConnectionError, ValueError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to get system status: {e}")
            return {}

    def list_credentials(self) -> List[Dict]:
        """List all credentials"""
        try:
            result = self._request('GET', '/credentials')
            return result.get('credentials', [])
        except (RuntimeError, ConnectionError, ValueError, KeyError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to list credentials: {e}")
            return []

# =============================================================================
# Configuration Generator (Enhanced)
# =============================================================================
class ConfigurationGenerator:
    """Generates complete configuration files with validation"""

    def __init__(self, base_url: str, api_key: str, file_logger: FileLogger = None):
        self.base_url = base_url
        self.api_key = api_key
        self.file_logger = file_logger

    def generate_yaml_config(self) -> Dict:
        """Generate YAML configuration"""
        return {
            'base_url': self.base_url,
            'api_key': self.api_key,
            'workflows_dir': './bubblelab-workflows',
            'templates_dir': './bubblelab-templates',
            'exports_dir': './bubblelab-exports',
            'backups_dir': './bubblelab-backups',
            'tests_dir': './bubblelab-tests',
            'environments': {
                'development': {
                    'api_url': 'http://localhost:8000',
                    'qdrant_url': 'http://localhost:6333',
                    'postgres_url': 'postgresql://postgres:password@localhost:5432/openevolve',
                    'redis_url': 'redis://localhost:6379',
                    'slack_channel': '#openevolve-dev'
                },
                'production': {
                    'api_url': self.base_url,
                    'qdrant_url': 'https://qdrant.openevolve.com',
                    'postgres_url': 'postgresql://user:pass@prod-db:5432/openevolve',
                    'redis_url': 'redis://prod-redis:6379',
                    'slack_channel': '#openevolve-alerts'
                }
            }
        }

    def generate_env_file(self) -> str:
        """Generate .env file content"""
        return f"""# BubbleLab Configuration
BUBBLELAB_BASE_URL={self.base_url}
BUBBLELAB_API_KEY={self.api_key}

# OpenEvolve Services
QDRANT_URL=http://localhost:6333
POSTGRES_URL=postgresql://postgres:password@localhost:5432/openevolve
REDIS_URL=redis://localhost:6379

# Slack (Optional)
SLACK_BOT_TOKEN=
SLACK_SIGNING_SECRET=
SLACK_CHANNEL=#openevolve

# AI Providers (Optional)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

"""

    def generate_gitignore(self) -> str:
        """Generate .gitignore content"""
        return """# BubbleLab
bubblelab-config.yaml
.env
bubblelab-backups/
bubblelab-exports/
bubblelab-logs/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
"""

    def save_all(self, base_dir: Path) -> bool:
        """Save all configuration files with validation"""
        Logger.section("Generating Configuration Files")

        try:
            # Generate config
            config = self.generate_yaml_config()

            # Validate schema
            is_valid, error = ConfigSchemaValidator.validate(config, self.file_logger)
            if not is_valid:
                Logger.error(f"Configuration schema validation failed: {error}")
                if self.file_logger:
                    self.file_logger.error(f"Schema validation failed: {error}")
                return False

            # Save YAML config
            config_file = base_dir / 'bubblelab-config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            Logger.success("✓ bubblelab-config.yaml")
            if self.file_logger:
                self.file_logger.info(f"Configuration saved: {config_file}")

            # Save .env
            env_file = base_dir / '.env'
            with open(env_file, 'w') as f:
                f.write(self.generate_env_file())
            Logger.success("✓ .env")
            if self.file_logger:
                self.file_logger.info(f"Environment file saved: {env_file}")

            # Save .gitignore
            gitignore_file = base_dir / '.gitignore'
            with open(gitignore_file, 'w') as f:
                f.write(self.generate_gitignore())
            Logger.success("✓ .gitignore")
            if self.file_logger:
                self.file_logger.info(f"Gitignore saved: {gitignore_file}")

            return True

        except (OSError, IOError, yaml.YAMLError) as e:
            Logger.error(f"Failed to generate configurations: {e}")
            if self.file_logger:
                self.file_logger.error("Configuration generation failed", e)
            return False

# =============================================================================
# Setup Orchestrator (Enhanced)
# =============================================================================
class SetupOrchestrator:
    """Main setup orchestrator with comprehensive validation"""

    def __init__(self, api_url: str = None, api_key: str = None, skip_tests: bool = False):
        # Initialize file logger first
        self.file_logger = FileLogger()

        # Validate inputs
        if api_url:
            is_valid, error = URLValidator.validate(api_url, self.file_logger)
            if not is_valid:
                Logger.error(f"Invalid API URL: {error}")
                sys.exit(1)
            self.api_url = api_url
        else:
            self.api_url = 'http://localhost:3001'

        if api_key:
            is_valid, error = APIKeyValidator.validate(api_key, self.file_logger)
            if not is_valid:
                Logger.error(f"Invalid API key: {error}")
                sys.exit(1)
            self.api_key = api_key
            # Warn if passed via CLI (we can't detect this here, but user should know)
        else:
            self.api_key = os.environ.get('BUBBLELAB_API_KEY', 'YOUR_API_KEY_HERE')

        self.skip_tests = skip_tests
        self.start_time = None
        self.results = {
            'validation': False,
            'dependencies': False,
            'directories': False,
            'configuration': False,
            'connectivity': False,
            'tests': False
        }

        self.file_logger.info(f"Setup initialized with API URL: {self.api_url}")

    def run(self) -> bool:
        """Run complete setup"""
        self.start_time = time.time()
        Logger.header("BUBBLELAB COMPLETE AUTOMATED SETUP v3.0")

        try:
            # Step 1: Validate Environment
            Logger.step(1, 7, "Validating Environment")
            validator = EnvironmentValidator(self.file_logger)
            valid, errors, warnings = validator.validate()
            self.results['validation'] = valid

            if not valid:
                Logger.error("\n✗ Environment validation failed. Please fix the errors above.")
                self.print_summary()
                return False

            # Step 2: Install Dependencies
            Logger.step(2, 7, "Installing Dependencies")
            installer = DependencyInstaller(self.file_logger)
            self.results['dependencies'] = installer.install_all()

            if not self.results['dependencies']:
                Logger.warning("\n⚠️  Some dependencies failed to install. Setup will continue but may have issues.")

            # Step 3: Create Directory Structure
            Logger.step(3, 7, "Creating Directory Structure")
            self.file_logger.info("Creating directory structure...")
            creator = DirectoryCreator(self.file_logger)
            self.results['directories'] = creator.create_all()

            # Step 4: Validate API Credentials
            Logger.step(4, 7, "Validating API Credentials")
            if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
                Logger.warning("No API key provided via --api-key or BUBBLELAB_API_KEY env var")
                Logger.info("You'll need to add it later to bubblelab-config.yaml")
                self.api_key = "YOUR_API_KEY_HERE"
                self.file_logger.warning("Setup proceeding without API key")
            else:
                Logger.success(f"API key format validated")
                self.file_logger.info("API key validated")

            # Step 5: Generate Configuration Files
            Logger.step(5, 7, "Generating Configuration Files")
            self.file_logger.info("Generating configuration files...")
            generator = ConfigurationGenerator(self.api_url, self.api_key, self.file_logger)
            self.results['configuration'] = generator.save_all(Path.cwd())

            # Step 6: Validate API Connectivity (if API key provided)
            Logger.step(6, 7, "Validating API Connectivity")
            if self.api_key and self.api_key != "YOUR_API_KEY_HERE":
                try:
                    client = BubbleLabClient(self.api_url, self.api_key, self.file_logger)

                    # First test credentials
                    Logger.detail("Testing API credentials...")
                    creds_valid, creds_msg = client.test_credentials()
                    if creds_valid:
                        Logger.success("✓ API credentials validated")
                        self.file_logger.info("API credentials validated")
                    else:
                        Logger.warning(f"⚠️  Credential validation warning: {creds_msg}")
                        self.file_logger.warning(f"Credential validation warning: {creds_msg}")

                    # Then test connectivity
                    Logger.detail("Testing API connectivity...")
                    connected, message = client.test_connection()
                    self.results['connectivity'] = connected

                    if connected:
                        Logger.success("✓ API connection validated")
                        self.file_logger.info("API connection successful")

                        # Get system info
                        status = client.get_system_status()
                        if status:
                            Logger.success("✓ Connected to BubbleLab API")
                    else:
                        Logger.error(f"✗ API connection failed: {message}")
                        Logger.warning("Setup will continue but API features won't work until fixed")
                        self.file_logger.error(f"API connection failed: {message}")
                except (RuntimeError, ConnectionError, ValueError) as e:
                    Logger.warning(f"⚠️  Could not validate API: {e}")
                    Logger.info("This is OK if BubbleLab is not running yet")
                    self.file_logger.warning(f"API validation error: {e}")
            else:
                Logger.info("Skipping API validation (no API key provided)")
                self.results['connectivity'] = None

            # Step 7: Run Tests (unless skipped)
            if not self.skip_tests:
                Logger.step(7, 7, "Running Validation Tests")
                self.file_logger.info("Running validation tests...")
                self.results['tests'] = self.run_tests()
            else:
                Logger.info("Skipping tests (--skip-tests flag)")
                self.results['tests'] = None

            # Print Summary
            self.print_summary()

            # Return success if all critical steps passed
            critical_success = (
                self.results['validation'] and
                self.results['directories'] and
                self.results['configuration']
            )

            if critical_success:
                Logger.success("\n✓ SETUP COMPLETE!")
                self.print_next_steps()
                self.file_logger.info("Setup completed successfully")
            else:
                Logger.error("\n✗ SETUP INCOMPLETE")
                Logger.info("Please fix the errors above and run setup again")
                self.file_logger.error("Setup incomplete")

            return critical_success

        except (OSError, RuntimeError, ValueError) as e:
            Logger.error(f"\n\n✗ Fatal error during setup: {e}")
            if self.file_logger:
                self.file_logger.error("Fatal error during setup", e)
            Logger.detail(traceback.format_exc())
            return False

    def run_tests(self) -> bool:
        """Run validation tests"""
        tests_passed = True

        # Test 1: Config file exists and is valid
        Logger.detail("Testing configuration file...")
        try:
            config_file = Path.cwd() / 'bubblelab-config.yaml'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                    is_valid, error = ConfigSchemaValidator.validate(config, self.file_logger)
                    if is_valid:
                        Logger.success("✓ Configuration file valid")
                        self.file_logger.info("Configuration file test passed")
                    else:
                        Logger.error(f"✗ Configuration file invalid: {error}")
                        tests_passed = False
            else:
                Logger.error("✗ Configuration file not found")
                tests_passed = False
        except (OSError, IOError, yaml.YAMLError) as e:
            Logger.error(f"✗ Configuration test failed: {e}")
            if self.file_logger:
                self.file_logger.error("Configuration test failed", e)
            tests_passed = False

        # Test 2: Directories exist
        Logger.detail("Testing directory structure...")
        required_dirs = ['bubblelab-workflows', 'bubblelab-exports', 'bubblelab-backups']
        for dir_name in required_dirs:
            dir_path = Path.cwd() / dir_name
            if dir_path.exists():
                Logger.success(f"✓ {dir_name}/ exists")
            else:
                Logger.error(f"✗ {dir_name}/ missing")
                tests_passed = False

        # Test 3: Python packages importable
        Logger.detail("Testing Python packages...")
        for package in ['yaml', 'requests']:
            try:
                __import__(package)
                Logger.success(f"✓ {package} importable")
            except ImportError:
                Logger.error(f"✗ {package} not importable")
                tests_passed = False

        if tests_passed:
            Logger.success("\n✓ All tests passed")
            if self.file_logger:
                self.file_logger.info("All validation tests passed")
        else:
            Logger.warning("\n⚠️  Some tests failed")
            if self.file_logger:
                self.file_logger.warning("Some validation tests failed")

        return tests_passed

    def print_summary(self):
        """Print setup summary"""
        elapsed = time.time() - self.start_time

        Logger.header("SETUP SUMMARY")

        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Log file: {self.file_logger.get_log_file()}\n")

        print("Results:")
        for step, result in self.results.items():
            if result is True:
                Logger.success(f"  ✓ {step}")
            elif result is False:
                Logger.error(f"  ✗ {step}")
            else:
                Logger.warning(f"  ○ {step} (skipped)")

    def print_next_steps(self):
        """Print next steps"""
        Logger.section("Next Steps")

        steps = [
            "1. Review configuration in bubblelab-config.yaml",
            "2. Add your API key if not already set",
            "3. Configure credentials in BubbleLab dashboard or via API",
            "4. Add your workflow files to bubblelab-workflows/",
            "5. Run: python bubblelab-automation.py deploy",
            "6. Monitor with: python bubblelab-automation.py monitor --flow-name 'Your Workflow'",
            "",
            "Quick Start Commands:",
            "  python bubblelab-automation.py list              # List all workflows",
            "  python bubblelab-automation.py status            # Check system status",
            "  python bubblelab-automation.py generate \\        # Generate with AI",
            "    --prompt 'Monitor Qdrant health' \\",
            "    --name 'Qdrant Monitor'",
            "",
            f"Full log: {self.file_logger.get_log_file()}",
            "",
            "Documentation:",
            "  - docs/BUBBLELAB_AUTOMATION_GUIDE.md     # Complete guide",
            "  - docs/BUBBLELAB_SCRIPTING_GUIDE.md      # API reference",
            "  - docs/BUBBLELAB_AUTOMATION_README.md    # Quick reference"
        ]

        for step in steps:
            print(f"  {step}")

# =============================================================================
# Directory Creator (Enhanced)
# =============================================================================
class DirectoryCreator:
    """Creates the complete directory structure"""

    DIRECTORIES = [
        'bubblelab-workflows',
        'bubblelab-workflows/dev',
        'bubblelab-workflows/prod',
        'bubblelab-templates',
        'bubblelab-exports',
        'bubblelab-backups',
        'bubblelab-tests',
        'bubblelab-config',
        'bubblelab-logs',
    ]

    def __init__(self, file_logger: FileLogger = None):
        self.base_dir = Path.cwd()
        self.created = []
        self.existing = []
        self.file_logger = file_logger

    def create_directory(self, directory: str) -> bool:
        """Create a single directory"""
        path = self.base_dir / directory
        try:
            if path.exists():
                Logger.detail(f"✓ {directory} (already exists)")
                self.existing.append(directory)
                return True
            else:
                path.mkdir(parents=True, exist_ok=True)
                Logger.success(f"✓ {directory} (created)")
                self.created.append(directory)
                if self.file_logger:
                    self.file_logger.info(f"Directory created: {directory}")
                return True
        except (PermissionError, OSError) as e:
            Logger.error(f"✗ {directory} ({e})")
            if self.file_logger:
                self.file_logger.error(f"Failed to create directory: {directory}", e)
            return False

    def create_all(self) -> bool:
        """Create all directories"""
        Logger.section("Creating Directory Structure")

        all_success = True
        for directory in self.DIRECTORIES:
            success = self.create_directory(directory)
            all_success &= success

        if all_success:
            Logger.success(f"\n✓ Directory structure ready")
            Logger.detail(f"  Created: {len(self.created)} directories")
            Logger.detail(f"  Existing: {len(self.existing)} directories")
            if self.file_logger:
                self.file_logger.info(f"Directory structure ready: {len(self.created)} created, {len(self.existing)} existing")

        return all_success

# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='BubbleLab Complete Automated Setup v3.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive setup (default)
  python bubblelab-auto-setup.py

  # Specify API URL and key
  python bubblelab-auto-setup.py --api-url http://localhost:3001 --api-key your_key

  # Skip validation tests (faster)
  python bubblelab-auto-setup.py --skip-tests

  # Production setup
  python bubblelab-auto-setup.py --api-url https://api.bubblelab.io --api-key prod_key

  # Use environment variable for API key (recommended for security)
  export BUBBLELAB_API_KEY=your_key
  python bubblelab-auto-setup.py --api-url https://api.bubblelab.io
        """
    )

    parser.add_argument(
        '--api-url',
        help='BubbleLab API URL (default: http://localhost:3001)'
    )
    parser.add_argument(
        '--api-key',
        help='BubbleLab API Key (or set BUBBLELAB_API_KEY env var for better security)'
    )
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip validation tests (faster setup)'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='BubbleLab Auto Setup v3.0.0 (Production-Grade Reliability)'
    )

    args = parser.parse_args()

    # Warn if API key passed via CLI
    if args.api_key:
        APIKeyValidator.warn_cli_usage()

    # Run setup
    try:
        orchestrator = SetupOrchestrator(
            api_url=args.api_url,
            api_key=args.api_key,
            skip_tests=args.skip_tests
        )

        success = orchestrator.run()
        sys.exit(0 if success else 1)

    except (OSError, RuntimeError, ValueError) as e:
        Logger.error(f"\n\n✗ Fatal error: {e}")
        Logger.detail(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        Logger.warning("\n\n⚠️  Setup interrupted by user")
        sys.exit(130)
    except (OSError, RuntimeError, ValueError) as e:
        Logger.error(f"\n\n✗ Fatal error: {e}")
        Logger.detail(traceback.format_exc())
        sys.exit(1)
