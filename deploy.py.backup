"""
Deployment Script for Sovereign-Grade Problem Decomposition System
Task 16.3: Build deployment automation
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


class DeploymentManager:
    """Manages deployment of the sovereign system."""
    
    def __init__(self, environment: str = "production"):
        """
        Initialize deployment manager.
        
        Args:
            environment: Target environment (development, staging, production)
        """
        self.environment = environment
        self.project_root = Path(__file__).parent
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load deployment configuration."""
        config_file = self.project_root / f"deploy_config_{self.environment}.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "python_version": "3.11",
            "required_packages": [
                "streamlit",
                "pytest",
                "psutil",
                "requests"
            ],
            "database_path": "sovereign_system.db",
            "log_level": "INFO",
            "health_check_port": 8000,
            "api_port": 8501
        }
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("Checking prerequisites...")
        
        # Check Python version
        python_version = (sys.version_info.major, sys.version_info.minor)
        required_version = self.config.get("python_version", "3.11")
        required_major, required_minor = map(int, required_version.split('.'))
        
        if python_version < (required_major, required_minor):
            print(f"❌ Python {required_version}+ required, found {python_version[0]}.{python_version[1]}")
            return False
        print(f"✓ Python {python_version[0]}.{python_version[1]}")
        
        # Check required files
        required_files = [
            "problem_analyzer.py",
            "decomposition_engine.py",
            "sovereign_gauntlets.py",
            "sovereign_quality_assessment.py",
            "sovereign_reliability.py"
        ]
        
        for file in required_files:
            if not (self.project_root / file).exists():
                print(f"❌ Required file missing: {file}")
                return False
        print(f"✓ All required files present")
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        print("\nInstalling dependencies...")
        
        try:
            # Install from requirements.txt if it exists
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                    check=True,
                    capture_output=True
                )
                print("✓ Dependencies installed from requirements.txt")
            else:
                # Install individual packages
                for package in self.config["required_packages"]:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", package],
                        check=True,
                        capture_output=True
                    )
                print(f"✓ Installed {len(self.config['required_packages'])} packages")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def initialize_database(self) -> bool:
        """Initialize the database."""
        print("\nInitializing database...")
        
        try:
            from sovereign_persistence import SovereignDatabase
            
            db = SovereignDatabase()
            db.init_database()
            
            print(f"✓ Database initialized at {self.config['database_path']}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize database: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run test suite."""
        print("\nRunning tests...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "test_sovereign*.py", "-v", "--tb=short"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Count passed tests
                output = result.stdout
                if "passed" in output:
                    print(f"✓ All tests passed")
                    return True
            
            print(f"❌ Some tests failed")
            print(result.stdout[-500:])  # Print last 500 chars
            return False
        except Exception as e:
            print(f"❌ Failed to run tests: {e}")
            return False
    
    def create_config_files(self) -> bool:
        """Create necessary configuration files."""
        print("\nCreating configuration files...")
        
        try:
            # Create .env file if it doesn't exist
            env_file = self.project_root / ".env"
            if not env_file.exists():
                with open(env_file, 'w') as f:
                    f.write(f"ENVIRONMENT={self.environment}\n")
                    f.write(f"LOG_LEVEL={self.config['log_level']}\n")
                    f.write(f"DATABASE_PATH={self.config['database_path']}\n")
                print("✓ Created .env file")
            
            # Create logging configuration
            log_config = {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "standard": {
                        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                    }
                },
                "handlers": {
                    "file": {
                        "class": "logging.FileHandler",
                        "filename": "sovereign_system.log",
                        "formatter": "standard"
                    },
                    "console": {
                        "class": "logging.StreamHandler",
                        "formatter": "standard"
                    }
                },
                "root": {
                    "level": self.config['log_level'],
                    "handlers": ["file", "console"]
                }
            }
            
            log_config_file = self.project_root / "logging_config.json"
            with open(log_config_file, 'w') as f:
                json.dump(log_config, f, indent=2)
            print("✓ Created logging configuration")
            
            return True
        except Exception as e:
            print(f"❌ Failed to create config files: {e}")
            return False
    
    def setup_health_monitoring(self) -> bool:
        """Setup health monitoring."""
        print("\nSetting up health monitoring...")
        
        try:
            from sovereign_reliability import get_health_monitor
            
            monitor = get_health_monitor()
            
            # Register basic health checks
            def check_database():
                try:
                    from sovereign_persistence import SovereignDatabase
                    db = SovereignDatabase()
                    return db.connection is not None
                except:
                    return False
            
            def check_memory():
                try:
                    import psutil
                    memory = psutil.virtual_memory()
                    return memory.percent < 90  # Less than 90% used
                except:
                    return True  # If can't check, assume OK
            
            monitor.register_check("database", check_database)
            monitor.register_check("memory", check_memory)
            
            # Run initial health check
            results = monitor.run_health_checks()
            
            if results['overall_healthy']:
                print("✓ Health monitoring configured and system healthy")
                return True
            else:
                print("⚠ Health monitoring configured but system unhealthy")
                return False
        except Exception as e:
            print(f"❌ Failed to setup health monitoring: {e}")
            return False
    
    def create_startup_script(self) -> bool:
        """Create startup script."""
        print("\nCreating startup script...")
        
        try:
            # Create bash/batch script based on OS
            if os.name == 'nt':  # Windows
                script_name = "start_sovereign.bat"
                script_content = f"""@echo off
echo Starting Sovereign-Grade Problem Decomposition System...
python -m streamlit run api_server.py --server.port {self.config['api_port']}
"""
            else:  # Unix-like
                script_name = "start_sovereign.sh"
                script_content = f"""#!/bin/bash
echo "Starting Sovereign-Grade Problem Decomposition System..."
python -m streamlit run api_server.py --server.port {self.config['api_port']}
"""
            
            script_path = self.project_root / script_name
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable on Unix
            if os.name != 'nt':
                os.chmod(script_path, 0o755)
            
            print(f"✓ Created startup script: {script_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to create startup script: {e}")
            return False
    
    def deploy(self, skip_tests: bool = False) -> bool:
        """
        Run complete deployment process.
        
        Args:
            skip_tests: Whether to skip running tests
            
        Returns:
            True if deployment successful
        """
        print(f"\n{'='*60}")
        print(f"DEPLOYING SOVEREIGN SYSTEM - {self.environment.upper()}")
        print(f"{'='*60}\n")
        
        steps = [
            ("Prerequisites", self.check_prerequisites),
            ("Dependencies", self.install_dependencies),
            ("Database", self.initialize_database),
            ("Configuration", self.create_config_files),
            ("Health Monitoring", self.setup_health_monitoring),
            ("Startup Script", self.create_startup_script),
        ]
        
        if not skip_tests:
            steps.insert(3, ("Tests", self.run_tests))
        
        for step_name, step_func in steps:
            if not step_func():
                print(f"\n❌ Deployment failed at step: {step_name}")
                return False
        
        print(f"\n{'='*60}")
        print("✓ DEPLOYMENT SUCCESSFUL")
        print(f"{'='*60}\n")
        print(f"Environment: {self.environment}")
        print(f"API Port: {self.config['api_port']}")
        print(f"Database: {self.config['database_path']}")
        print(f"\nTo start the system, run:")
        if os.name == 'nt':
            print("  start_sovereign.bat")
        else:
            print("  ./start_sovereign.sh")
        print()
        
        return True
    
    def rollback(self) -> bool:
        """Rollback deployment."""
        print("\nRolling back deployment...")
        
        try:
            # Remove created files
            files_to_remove = [
                ".env",
                "logging_config.json",
                "start_sovereign.bat",
                "start_sovereign.sh",
                self.config['database_path']
            ]
            
            for file in files_to_remove:
                file_path = self.project_root / file
                if file_path.exists():
                    file_path.unlink()
                    print(f"✓ Removed {file}")
            
            print("✓ Rollback complete")
            return True
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return False


def main():
    """Main deployment entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy Sovereign-Grade Problem Decomposition System"
    )
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        default="production",
        help="Target environment"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback deployment"
    )
    
    args = parser.parse_args()
    
    manager = DeploymentManager(environment=args.environment)
    
    if args.rollback:
        success = manager.rollback()
    else:
        success = manager.deploy(skip_tests=args.skip_tests)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
