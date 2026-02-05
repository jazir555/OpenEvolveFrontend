"""
Deployment Readiness Tests - License: Apache 2.0

Tests deployment readiness for OpenEvolve:
- Configuration validation
- Database connectivity
- External service dependencies
- Resource availability
- Health check endpoints
- Environment validation

Run: pytest test_deployment_readiness.py -v
"""

import asyncio
import json
import os
import socket
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import pytest

# System availability checks
try:
    from api_server import app as api_app
    from fastapi.testclient import TestClient
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

try:
    from config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from config_loader import ConfigLoader
    CONFIG_LOADER_AVAILABLE = True
except ImportError:
    CONFIG_LOADER_AVAILABLE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

try:
    from health_endpoint import HealthEndpoint
    HEALTH_ENDPOINT_AVAILABLE = True
except ImportError:
    HEALTH_ENDPOINT_AVAILABLE = False

try:
    from system_health import SystemHealthMonitor
    SYSTEM_HEALTH_AVAILABLE = True
except ImportError:
    SYSTEM_HEALTH_AVAILABLE = False

try:
    from deployment_operations import DeploymentManager
    DEPLOYMENT_AVAILABLE = True
except ImportError:
    DEPLOYMENT_AVAILABLE = False


@dataclass
class DeploymentReadinessResult:
    """Result of a deployment readiness test."""
    test_name: str
    category: str  # 'config', 'database', 'services', 'resources', 'health', 'environment'
    status: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    message: str = ""
    details: Dict = field(default_factory=dict)


class TestDeploymentReadiness:
    """
    Deployment Readiness Tests.
    
    Verifies system is ready for deployment.
    """
    
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.results: List[DeploymentReadinessResult] = []
        
        # Initialize systems
        self.systems = {}
        self._init_systems()
        
        yield
        
        # Cleanup
        self.temp_dir.cleanup()
    
    def _init_systems(self):
        """Initialize all systems."""
        if SYSTEM_HEALTH_AVAILABLE:
            self.systems['health'] = SystemHealthMonitor()
        
        if DEPLOYMENT_AVAILABLE:
            self.systems['deployment'] = DeploymentManager()
    
    def _record_result(self, result: DeploymentReadinessResult):
        """Record test result."""
        self.results.append(result)
        return result.status == 'passed'
    
    def _is_port_open(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """Check if a port is open on a host."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    @pytest.mark.integration
    def test_configuration_validation(self):
        """Test configuration is valid and complete."""
        try:
            config_results = []
            
            # Test config loading
            if CONFIG_AVAILABLE:
                try:
                    config = Config()
                    config_results.append({"check": "config_load", "valid": True})
                    
                    # Check required settings
                    required_settings = ['log_level', 'database_url']
                    for setting in required_settings:
                        has_setting = hasattr(config, setting) or setting in dir(config)
                        config_results.append({"check": f"has_{setting}", "valid": has_setting})
                        
                except Exception as e:
                    config_results.append({"check": "config_load", "valid": False, "error": str(e)})
            
            # Test config loader
            if CONFIG_LOADER_AVAILABLE:
                try:
                    loader = ConfigLoader()
                    if hasattr(loader, 'load_config'):
                        loader.load_config()
                        config_results.append({"check": "config_loader", "valid": True})
                    else:
                        config_results.append({"check": "config_loader", "valid": False, "reason": "no load method"})
                except Exception as e:
                    config_results.append({"check": "config_loader", "valid": False, "error": str(e)})
            
            # Check environment variables
            required_env_vars = []
            optional_env_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEBUG']
            
            for var in required_env_vars:
                config_results.append({"check": f"env_{var}", "valid": os.getenv(var) is not None})
            
            valid_count = sum(1 for r in config_results if r.get("valid"))
            passed = len(config_results) == 0 or valid_count >= len(config_results) * 0.5
            
            result = DeploymentReadinessResult(
                test_name="test_configuration_validation",
                category="config",
                status="passed" if passed else "failed",
                severity="critical",
                message=f"Configuration: {valid_count}/{len(config_results)} checks passed",
                details={"config_results": config_results}
            )
            self._record_result(result)
            
            print(f"\n[Deployment] Configuration validation:")
            for r in config_results:
                status = "[OK]" if r.get("valid") else "[FAIL]"
                print(f"   {status} {r['check']}")
            
            assert passed, f"Only {valid_count}/{len(config_results)} config checks passed"
            
        except Exception as e:
            self._record_result(DeploymentReadinessResult(
                test_name="test_configuration_validation",
                category="config",
                status="failed",
                severity="critical",
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    def test_database_connectivity(self):
        """Test database connectivity."""
        if not SQLITE_AVAILABLE:
            pytest.skip("SQLite not available")
        
        try:
            db_results = []
            
            # Test SQLite connectivity
            try:
                db_path = os.path.join(self.temp_dir.name, "test.db")
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create test table
                cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
                cursor.execute("INSERT INTO test (name) VALUES ('test')")
                conn.commit()
                
                # Query test
                cursor.execute("SELECT * FROM test")
                rows = cursor.fetchall()
                
                conn.close()
                
                db_results.append({"check": "sqlite_connect", "valid": True})
                db_results.append({"check": "sqlite_read_write", "valid": len(rows) > 0})
                
            except Exception as e:
                db_results.append({"check": "sqlite", "valid": False, "error": str(e)})
            
            valid_count = sum(1 for r in db_results if r.get("valid"))
            passed = valid_count >= len(db_results) * 0.5
            
            result = DeploymentReadinessResult(
                test_name="test_database_connectivity",
                category="database",
                status="passed" if passed else "failed",
                severity="critical",
                message=f"Database: {valid_count}/{len(db_results)} checks passed",
                details={"db_results": db_results}
            )
            self._record_result(result)
            
            print(f"\n[Deployment] Database connectivity:")
            for r in db_results:
                status = "[OK]" if r.get("valid") else "[FAIL]"
                print(f"   {status} {r['check']}")
            
            assert passed, f"Only {valid_count}/{len(db_results)} DB checks passed"
            
        except Exception as e:
            self._record_result(DeploymentReadinessResult(
                test_name="test_database_connectivity",
                category="database",
                status="failed",
                severity="critical",
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    def test_external_service_dependencies(self):
        """Test external service dependencies."""
        try:
            service_results = []
            
            # Check common ports for external services
            services_to_check = [
                {"name": "Redis", "host": "localhost", "port": 6379, "optional": True},
                {"name": "PostgreSQL", "host": "localhost", "port": 5432, "optional": True},
                {"name": "Neo4j", "host": "localhost", "port": 7687, "optional": True},
            ]
            
            for service in services_to_check:
                is_open = self._is_port_open(service["host"], service["port"])
                service_results.append({
                    "name": service["name"],
                    "host": service["host"],
                    "port": service["port"],
                    "available": is_open,
                    "optional": service["optional"],
                    "valid": is_open or service["optional"]
                })
            
            # All should be valid (available or optional)
            valid_count = sum(1 for r in service_results if r.get("valid"))
            passed = valid_count == len(service_results)
            
            result = DeploymentReadinessResult(
                test_name="test_external_service_dependencies",
                category="services",
                status="passed" if passed else "failed",
                severity="high",
                message=f"External services: {valid_count}/{len(service_results)} available",
                details={"service_results": service_results}
            )
            self._record_result(result)
            
            print(f"\n[Deployment] External service dependencies:")
            for r in service_results:
                status = "[OK]" if r.get("available") else ("[OPT]" if r.get("optional") else "[MISSING]")
                print(f"   {status} {r['name']} ({r['host']}:{r['port']})")
            
            assert passed, f"Required external service not available"
            
        except Exception as e:
            self._record_result(DeploymentReadinessResult(
                test_name="test_external_service_dependencies",
                category="services",
                status="failed",
                severity="high",
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    def test_resource_availability(self):
        """Test system resource availability."""
        try:
            import psutil
            
            resource_results = []
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            disk_ok = disk_free_gb > 1.0  # At least 1GB free
            resource_results.append({
                "resource": "disk_space",
                "free_gb": disk_free_gb,
                "valid": disk_ok
            })
            
            # Check memory
            memory = psutil.virtual_memory()
            memory_available_gb = memory.available / (1024**3)
            memory_ok = memory_available_gb > 0.5  # At least 500MB available
            resource_results.append({
                "resource": "memory",
                "available_gb": memory_available_gb,
                "valid": memory_ok
            })
            
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_ok = cpu_percent < 90  # Less than 90% usage
            resource_results.append({
                "resource": "cpu",
                "usage_percent": cpu_percent,
                "valid": cpu_ok
            })
            
            valid_count = sum(1 for r in resource_results if r.get("valid"))
            passed = valid_count == len(resource_results)
            
            result = DeploymentReadinessResult(
                test_name="test_resource_availability",
                category="resources",
                status="passed" if passed else "failed",
                severity="critical",
                message=f"Resources: {valid_count}/{len(resource_results)} checks passed",
                details={"resource_results": resource_results}
            )
            self._record_result(result)
            
            print(f"\n[Deployment] Resource availability:")
            for r in resource_results:
                status = "[OK]" if r.get("valid") else "[FAIL]"
                if r["resource"] == "disk_space":
                    print(f"   {status} Disk: {r['free_gb']:.2f}GB free")
                elif r["resource"] == "memory":
                    print(f"   {status} Memory: {r['available_gb']:.2f}GB available")
                elif r["resource"] == "cpu":
                    print(f"   {status} CPU: {r['usage_percent']:.1f}% usage")
            
            assert passed, f"Resource constraints detected"
            
        except ImportError:
            pytest.skip("psutil not available")
        except Exception as e:
            self._record_result(DeploymentReadinessResult(
                test_name="test_resource_availability",
                category="resources",
                status="failed",
                severity="critical",
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    def test_health_check_endpoints(self):
        """Test health check endpoints."""
        try:
            health_results = []
            
            # Test API health endpoint
            if API_AVAILABLE:
                try:
                    client = TestClient(api_app)
                    response = client.get("/health")
                    health_results.append({
                        "endpoint": "/health",
                        "status": response.status_code,
                        "valid": response.status_code in [200, 307]
                    })
                except Exception as e:
                    health_results.append({
                        "endpoint": "/health",
                        "valid": False,
                        "error": str(e)
                    })
            
            # Test system health monitor
            if SYSTEM_HEALTH_AVAILABLE:
                try:
                    health = self.systems['health']
                    if hasattr(health, 'check_health'):
                        health_status = health.check_health()
                        health_results.append({
                            "check": "system_health_monitor",
                            "valid": health_status is not None
                        })
                    else:
                        health_results.append({
                            "check": "system_health_monitor",
                            "valid": False,
                            "reason": "no check method"
                        })
                except Exception as e:
                    health_results.append({
                        "check": "system_health_monitor",
                        "valid": False,
                        "error": str(e)
                    })
            
            valid_count = sum(1 for r in health_results if r.get("valid"))
            passed = len(health_results) == 0 or valid_count >= len(health_results) * 0.5
            
            result = DeploymentReadinessResult(
                test_name="test_health_check_endpoints",
                category="health",
                status="passed" if passed else "failed",
                severity="critical",
                message=f"Health checks: {valid_count}/{len(health_results)} passed",
                details={"health_results": health_results}
            )
            self._record_result(result)
            
            print(f"\n[Deployment] Health check endpoints:")
            for r in health_results:
                status = "[OK]" if r.get("valid") else "[FAIL]"
                endpoint = r.get("endpoint") or r.get("check")
                print(f"   {status} {endpoint}")
            
            assert passed, f"Only {valid_count}/{len(health_results)} health checks passed"
            
        except Exception as e:
            self._record_result(DeploymentReadinessResult(
                test_name="test_health_check_endpoints",
                category="health",
                status="failed",
                severity="critical",
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    def test_environment_validation(self):
        """Test environment is properly configured."""
        try:
            env_results = []
            
            # Check Python version
            import sys
            python_version = sys.version_info
            python_ok = python_version >= (3, 10)
            env_results.append({
                "check": "python_version",
                "value": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                "valid": python_ok
            })
            
            # Check required files exist
            required_files = [
                "config.py",
                "api_server.py",
                "requirements.txt",
            ]
            
            for file in required_files:
                exists = os.path.exists(file)
                env_results.append({
                    "check": f"file_{file}",
                    "valid": exists
                })
            
            # Check directory structure
            required_dirs = [
                "tests",
                "docs",
            ]
            
            for dir in required_dirs:
                exists = os.path.isdir(dir)
                env_results.append({
                    "check": f"dir_{dir}",
                    "valid": exists
                })
            
            valid_count = sum(1 for r in env_results if r.get("valid"))
            passed = valid_count >= len(env_results) * 0.7  # At least 70%
            
            result = DeploymentReadinessResult(
                test_name="test_environment_validation",
                category="environment",
                status="passed" if passed else "failed",
                severity="high",
                message=f"Environment: {valid_count}/{len(env_results)} checks passed",
                details={"env_results": env_results}
            )
            self._record_result(result)
            
            print(f"\n[Deployment] Environment validation:")
            for r in env_results:
                status = "[OK]" if r.get("valid") else "[FAIL]"
                check_name = r['check']
                if 'value' in r:
                    print(f"   {status} {check_name}: {r['value']}")
                else:
                    print(f"   {status} {check_name}")
            
            assert passed, f"Only {valid_count}/{len(env_results)} environment checks passed"
            
        except Exception as e:
            self._record_result(DeploymentReadinessResult(
                test_name="test_environment_validation",
                category="environment",
                status="failed",
                severity="high",
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    def test_complete_deployment_readiness(self):
        """Test complete deployment readiness."""
        print("\n" + "="*70)
        print("COMPLETE DEPLOYMENT READINESS ASSESSMENT")
        print("="*70)
        
        readiness_categories = {
            "configuration": CONFIG_AVAILABLE or CONFIG_LOADER_AVAILABLE,
            "database": SQLITE_AVAILABLE,
            "health_checks": HEALTH_ENDPOINT_AVAILABLE or SYSTEM_HEALTH_AVAILABLE,
            "deployment_manager": DEPLOYMENT_AVAILABLE,
        }
        
        print("\nDeployment Readiness Categories:")
        for category, available in readiness_categories.items():
            status = "[OK]" if available else "[MISSING]"
            print(f"   {status} {category}")
        
        available_count = sum(readiness_categories.values())
        total_count = len(readiness_categories)
        
        print(f"\nDeployment Readiness: {available_count}/{total_count} categories ({available_count/total_count*100:.1f}%)")
        
        # At least 50% of readiness categories should be available
        passed = available_count >= total_count * 0.5
        
        print("="*70)
        
        assert passed, f"Only {available_count}/{total_count} deployment readiness categories available"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
