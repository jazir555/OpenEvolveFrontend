"""
Z3 Prover Service Bubble - Deployment Script

Deploys the Z3 Service Bubble with all components:
- Service startup
- Health checks
- Monitoring setup
- Load balancing configuration

Usage:
    python deploy_z3_service.py --environment production
    python deploy_z3_service.py --environment development --port 8765

Author: OpenEvolve
Created: 2026-02-04
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Z3ServiceDeployer:
    """Deploy Z3 Prover Service Bubble."""
    
    def __init__(self, environment: str = "development", port: int = 8765):
        self.environment = environment
        self.port = port
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        configs = {
            "development": {
                "host": "127.0.0.1",
                "port": self.port,
                "workers": 1,
                "reload": True,
                "log_level": "debug",
                "timeout": 300.0,
                "cache_enabled": True,
                "monitoring_enabled": True,
            },
            "staging": {
                "host": "0.0.0.0",
                "port": self.port,
                "workers": 2,
                "reload": False,
                "log_level": "info",
                "timeout": 60.0,
                "cache_enabled": True,
                "monitoring_enabled": True,
            },
            "production": {
                "host": "0.0.0.0",
                "port": self.port,
                "workers": 4,
                "reload": False,
                "log_level": "warning",
                "timeout": 30.0,
                "cache_enabled": True,
                "monitoring_enabled": True,
            }
        }
        return configs.get(self.environment, configs["development"])
    
    def check_dependencies(self) -> bool:
        """Check all required dependencies."""
        logger.info("Checking dependencies...")
        
        checks = {
            "python": self._check_python(),
            "z3": self._check_z3(),
            "fastapi": self._check_fastapi(),
        }
        
        all_passed = all(checks.values())
        
        for name, passed in checks.items():
            status = "OK" if passed else "FAIL"
            logger.info(f"  {name}: {status}")
        
        return all_passed
    
    def _check_python(self) -> bool:
        """Check Python version."""
        version = sys.version_info
        return version.major == 3 and version.minor >= 10
    
    def _check_z3(self) -> bool:
        """Check Z3 installation."""
        try:
            import z3
            return True
        except ImportError:
            pass
        
        # Check for Z3 binary
        try:
            result = subprocess.run(
                ['z3', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_fastapi(self) -> bool:
        """Check FastAPI installation."""
        try:
            import fastapi
            return True
        except ImportError:
            return False
    
    def setup_environment(self) -> bool:
        """Setup environment variables."""
        logger.info("Setting up environment...")
        
        env_vars = {
            "Z3_SERVER_HOST": self.config["host"],
            "Z3_SERVER_PORT": str(self.config["port"]),
            "Z3_TIMEOUT": str(self.config["timeout"]),
            "Z3_CACHE_ENABLED": str(self.config["cache_enabled"]),
            "Z3_MONITORING_ENABLED": str(self.config["monitoring_enabled"]),
            "Z3_ENVIRONMENT": self.environment,
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"  {key}={value}")
        
        return True
    
    def initialize_services(self) -> bool:
        """Initialize all service components."""
        logger.info("Initializing services...")
        
        try:
            # Import and initialize
            from z3_api_server import get_service_bubble
            
            bubble = get_service_bubble()
            status = bubble.get_status()
            
            logger.info(f"  Z3 available: {status.get('z3_available', False)}")
            logger.info(f"  Cache available: {status.get('cache_available', False)}")
            logger.info(f"  Monitor available: {status.get('monitor_available', False)}")
            
            return True
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            return False
    
    def start_server(self) -> bool:
        """Start the API server."""
        logger.info(f"Starting server on {self.config['host']}:{self.config['port']}...")
        
        try:
            import uvicorn
            
            uvicorn.run(
                "z3_api_server:app",
                host=self.config["host"],
                port=self.config["port"],
                workers=self.config["workers"],
                reload=self.config["reload"],
                log_level=self.config["log_level"]
            )
            
            return True
        except Exception as e:
            logger.error(f"Server startup failed: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        logger.info("Performing health check...")
        
        try:
            import requests
            
            url = f"http://{self.config['host']}:{self.config['port']}/health"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def deploy(self) -> bool:
        """Run full deployment."""
        logger.info("=" * 70)
        logger.info("Z3 PROVER SERVICE BUBBLE - DEPLOYMENT")
        logger.info("=" * 70)
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Port: {self.port}")
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("Dependency check failed!")
            return False
        
        # Setup environment
        if not self.setup_environment():
            logger.error("Environment setup failed!")
            return False
        
        # Initialize services
        if not self.initialize_services():
            logger.error("Service initialization failed!")
            return False
        
        # Start server
        logger.info("=" * 70)
        logger.info("DEPLOYMENT SUCCESSFUL - STARTING SERVER")
        logger.info("=" * 70)
        
        return self.start_server()


def create_docker_files():
    """Create Docker deployment files."""
    
    # Dockerfile
    dockerfile = '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    z3 \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8765/health')" || exit 1

# Start server
CMD ["python", "deploy_z3_service.py", "--environment", "production"]
'''
    
    with open("Dockerfile.z3", "w") as f:
        f.write(dockerfile)
    
    # docker-compose.yml
    compose = '''version: '3.8'

services:
  z3-service:
    build:
      context: .
      dockerfile: Dockerfile.z3
    ports:
      - "8765:8765"
    environment:
      - Z3_ENVIRONMENT=production
      - Z3_SERVER_HOST=0.0.0.0
      - Z3_SERVER_PORT=8765
      - Z3_CACHE_ENABLED=true
      - Z3_MONITORING_ENABLED=true
    volumes:
      - z3-cache:/app/cache
      - z3-logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 512M

volumes:
  z3-cache:
  z3-logs:
'''
    
    with open("docker-compose.z3.yml", "w") as f:
        f.write(compose)
    
    print("Created Dockerfile.z3 and docker-compose.z3.yml")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy Z3 Prover Service Bubble"
    )
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        default="development",
        help="Deployment environment"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Server port"
    )
    parser.add_argument(
        "--create-docker",
        action="store_true",
        help="Create Docker deployment files"
    )
    
    args = parser.parse_args()
    
    if args.create_docker:
        create_docker_files()
        return
    
    deployer = Z3ServiceDeployer(
        environment=args.environment,
        port=args.port
    )
    
    success = deployer.deploy()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
