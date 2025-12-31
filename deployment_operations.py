"""
Sovereign-Grade Problem Decomposition System - Deployment and Operations
Automated deployment scripts, backup/restore, and configuration management.
"""

import os
import sys
import json
import yaml
import subprocess
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import sqlite3
import zipfile
import tarfile
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for deployment operations"""
    environment: str
    database_path: str
    backup_path: str
    log_path: str
    api_keys: Dict[str, str]
    openevolve_config: Dict[str, Any]
    security_config: Dict[str, Any]


class DeploymentManager:
    """Manages deployment operations for the system"""
    
    def __init__(self, config_path: str = "deploy_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> DeploymentConfig:
        """Load deployment configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            # Create default configuration
            config_data = {
                'development': {
                    'database_path': 'sovereign_decomposition.db',
                    'backup_path': './backups',
                    'log_path': './logs',
                    'api_keys': {},
                    'openevolve_config': {},
                    'security_config': {
                        'encryption_key': None,
                        'jwt_secret': None
                    }
                }
            }
            # Save default config
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f)
        
        env_config = config_data.get(self._get_environment(), config_data['development'])
        return DeploymentConfig(
            environment=self._get_environment(),
            database_path=env_config.get('database_path', 'sovereign_decomposition.db'),
            backup_path=env_config.get('backup_path', './backups'),
            log_path=env_config.get('log_path', './logs'),
            api_keys=env_config.get('api_keys', {}),
            openevolve_config=env_config.get('openevolve_config', {}),
            security_config=env_config.get('security_config', {})
        )
    
    def _get_environment(self) -> str:
        """Get current environment"""
        return os.getenv('SOVEREIGN_ENV', 'development')
    
    def _setup_logging(self):
        """Setup logging for deployment operations"""
        os.makedirs(self.config.log_path, exist_ok=True)
        log_file = os.path.join(self.config.log_path, f"deployment_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def deploy(self, environment: str = None) -> bool:
        """Deploy the system to specified environment"""
        try:
            target_env = environment or self.config.environment
            logger.info(f"Starting deployment to {target_env} environment")
            
            # Pre-deployment validation
            if not self._pre_deployment_check():
                logger.error("Pre-deployment validation failed")
                return False
            
            # Backup current deployment (if exists)
            backup_path = self.backup_current_deployment()
            if not backup_path:
                logger.warning("Could not create backup before deployment")
            
            # Apply environment-specific configurations
            self._apply_environment_config(target_env)
            
            # Run database migrations
            migration_success = self._run_migrations()
            if not migration_success:
                logger.error("Database migrations failed")
                if backup_path:
                    self._restore_from_backup(backup_path)
                return False
            
            # Initialize services
            if not self._initialize_services():
                logger.error("Service initialization failed")
                if backup_path:
                    self._restore_from_backup(backup_path)
                return False
            
            # Post-deployment validation
            if not self._post_deployment_check():
                logger.error("Post-deployment validation failed")
                if backup_path:
                    self._restore_from_backup(backup_path)
                return False
            
            logger.info(f"Successfully deployed to {target_env} environment")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _pre_deployment_check(self) -> bool:
        """Perform pre-deployment validation"""
        logger.info("Running pre-deployment checks...")
        
        # Check required files
        required_files = [
            'problem_analyzer.py',
            'decomposition_engine.py', 
            'sovereign_team_coordination.py',
            'sovereign_solution_orchestration.py'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                logger.error(f"Required file missing: {file}")
                return False
        
        # Check for valid configuration
        if not self.config.database_path:
            logger.error("Database path not configured")
            return False
        
        logger.info("Pre-deployment checks passed")
        return True
    
    def _apply_environment_config(self, environment: str):
        """Apply environment-specific configuration"""
        logger.info(f"Applying configuration for {environment} environment")
        
        # Set environment variables
        os.environ['SOVEREIGN_ENV'] = environment
        os.environ['DATABASE_URL'] = f"sqlite:///{self.config.database_path}"
        
        # Set API keys from config if available
        for key_name, key_value in self.config.api_keys.items():
            os.environ[key_name] = key_value
        
        # Set OpenEvolve configuration
        if self.config.openevolve_config:
            for key, value in self.config.openevolve_config.items():
                os.environ[f"OPENEVOLVE_{key.upper()}"] = str(value)
    
    def _run_migrations(self) -> bool:
        """Run database migrations"""
        logger.info("Running database migrations...")
        
        try:
            # Import and run migrations
            from migrations import MIGRATIONS
            from sovereign_persistence import SovereignDatabase
            
            db = SovereignDatabase(self.config.database_path)
            db.apply_migrations()
            
            logger.info("Database migrations completed successfully")
            return True
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def _initialize_services(self) -> bool:
        """Initialize system services"""
        logger.info("Initializing services...")
        
        try:
            # Initialize authentication system
            from auth_system import get_auth_system
            auth_system = get_auth_system()
            
            # Initialize performance optimizer
            from performance_optimization import get_performance_optimizer
            perf_optimizer = get_performance_optimizer()
            perf_optimizer.optimize_database()
            
            # Initialize monitoring
            from monitoring_system import get_observability_manager
            obs_manager = get_observability_manager()
            
            logger.info("Services initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            return False
    
    def _post_deployment_check(self) -> bool:
        """Perform post-deployment validation"""
        logger.info("Running post-deployment checks...")
        
        # Verify database connectivity
        try:
            from sovereign_persistence import SovereignDatabase
            db = SovereignDatabase(self.config.database_path)
            # Try a simple operation
            db.list_problems(limit=1)
        except Exception as e:
            logger.error(f"Database connectivity test failed: {e}")
            return False
        
        logger.info("Post-deployment checks passed")
        return True
    
    def backup_current_deployment(self) -> Optional[str]:
        """Create a backup of the current deployment"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"sovereign_backup_{timestamp}.tar.gz"
            backup_path = os.path.join(self.config.backup_path, backup_filename)
            
            os.makedirs(self.config.backup_path, exist_ok=True)
            
            # Create backup archive
            with tarfile.open(backup_path, 'w:gz') as tar:
                # Add database file
                if os.path.exists(self.config.database_path):
                    tar.add(self.config.database_path, arcname='database.db')
                
                # Add configuration files
                config_dir = os.path.dirname(self.config_path)
                if os.path.exists(config_dir):
                    tar.add(config_dir, arcname='config')
                
                # Add any other important files
                for file_pattern in ['*.py', '*.md', 'requirements.txt']:
                    for file_path in Path('.').glob(file_pattern):
                        if str(file_path) != backup_path:  # Exclude backup file itself
                            tar.add(file_path, arcname=file_path)
            
            logger.info(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None
    
    def _restore_from_backup(self, backup_path: str) -> bool:
        """Restore from a backup"""
        try:
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(path='/tmp/sovereign_restore')
            
            # Copy database back
            backup_db = '/tmp/sovereign_restore/database.db'
            if os.path.exists(backup_db):
                shutil.copy2(backup_db, self.config.database_path)
                logger.info("Database restored from backup")
            
            # Copy config back
            backup_config = '/tmp/sovereign_restore/config'
            if os.path.exists(backup_config):
                shutil.copytree(backup_config, os.path.dirname(self.config_path), dirs_exist_ok=True)
                logger.info("Configuration restored from backup")
            
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False


class BackupManager:
    """Manages backup and restore operations"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.backup_dir = Path(config.backup_path)
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, backup_name: str = None, include_logs: bool = False) -> Optional[str]:
        """Create a system backup"""
        try:
            backup_name = backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backup_dir / f"{backup_name}.zip"
            
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
                # Add database
                if os.path.exists(self.config.database_path):
                    backup_zip.write(self.config.database_path, 'database.db')
                
                # Add configuration
                if os.path.exists(self.config_path):
                    backup_zip.write(self.config_path, 'deployment_config.yaml')
                
                # Add logs if requested
                if include_logs and os.path.exists(self.config.log_path):
                    for log_file in Path(self.config.log_path).glob("*.log"):
                        backup_zip.write(
                            log_file, 
                            f"logs/{log_file.name}"
                        )
                
                # Add important system files
                files_to_backup = [
                    'sovereign_data_models.py',
                    'sovereign_persistence.py',
                    'problem_analyzer.py',
                    'decomposition_engine.py',
                    'sovereign_team_coordination.py',
                    'sovereign_solution_orchestration.py'
                ]
                
                for file_path in files_to_backup:
                    if os.path.exists(file_path):
                        backup_zip.write(file_path, file_path)
            
            logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None
    
    def restore_backup(self, backup_path: str) -> bool:
        """Restore from a backup"""
        try:
            with zipfile.ZipFile(backup_path, 'r') as backup_zip:
                # Extract to temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    backup_zip.extractall(temp_dir)
                    
                    temp_path = Path(temp_dir)
                    
                    # Restore database
                    db_file = temp_path / 'database.db'
                    if db_file.exists():
                        shutil.copy2(db_file, self.config.database_path)
                        logger.info("Database restored")
                    
                    # Restore configuration
                    config_file = temp_path / 'deployment_config.yaml'
                    if config_file.exists():
                        shutil.copy2(config_file, self.config_path)
                        logger.info("Configuration restored")
                    
                    # Restore logs if present
                    logs_dir = temp_path / 'logs'
                    if logs_dir.exists():
                        shutil.copytree(logs_dir, Path(self.config.log_path), dirs_exist_ok=True)
                        logger.info("Logs restored")
            
            logger.info(f"Backup restored from: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Backup restore failed: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        for backup_file in self.backup_dir.glob("*.zip"):
            stat = backup_file.stat()
            backups.append({
                'name': backup_file.name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'path': str(backup_file)
            })
        
        # Sort by modification time (newest first)
        backups.sort(key=lambda x: x['modified'], reverse=True)
        return backups
    
    def cleanup_old_backups(self, days_to_keep: int = 30) -> int:
        """Remove backups older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        
        for backup in self.list_backups():
            if backup['modified'] < cutoff_date:
                os.remove(backup['path'])
                deleted_count += 1
                logger.info(f"Deleted old backup: {backup['name']}")
        
        logger.info(f"Cleaned up {deleted_count} old backups")
        return deleted_count


class WorkflowMigrationManager:
    """Manages migration of workflows between environments"""
    
    def __init__(self, source_db_path: str, target_db_path: str):
        self.source_db_path = source_db_path
        self.target_db_path = target_db_path
    
    def export_workflow(self, workflow_id: str, export_path: str) -> bool:
        """Export a workflow to a file"""
        try:
            # Connect to source database
            with sqlite3.connect(self.source_db_path) as source_conn:
                source_conn.row_factory = sqlite3.Row
                cursor = source_conn.cursor()
                
                # Get workflow data
                workflow_data = {}
                
                # Export problems
                cursor.execute("SELECT * FROM problems WHERE id = ?", (workflow_id,))
                problems = [dict(row) for row in cursor.fetchall()]
                workflow_data['problems'] = problems
                
                # Export decomposition plans related to these problems
                problem_ids = [p['id'] for p in problems]
                if problem_ids:
                    placeholders = ','.join(['?' for _ in problem_ids])
                    cursor.execute(f"""
                        SELECT * FROM decomposition_plans 
                        WHERE problem_id IN ({placeholders})
                    """, problem_ids)
                    plans = [dict(row) for row in cursor.fetchall()]
                    workflow_data['decomposition_plans'] = plans
                
                # Export sub-problems
                plan_ids = [p['id'] for p in plans] if 'plans' in locals() else []
                if plan_ids:
                    placeholders = ','.join(['?' for _ in plan_ids])
                    cursor.execute(f"""
                        SELECT * FROM sub_problems 
                        WHERE parent_id IN ({placeholders})
                        OR id IN (SELECT id FROM sub_problems WHERE parent_id IN ({placeholders}))
                    """, plan_ids + plan_ids)  # Double the plan_ids for the OR condition
                    sub_problems = [dict(row) for row in cursor.fetchall()]
                    workflow_data['sub_problems'] = sub_problems
                
                # Export solution attempts
                if 'sub_problems' in workflow_data:
                    sub_problem_ids = [sp['id'] for sp in workflow_data['sub_problems']]
                    if sub_problem_ids:
                        placeholders = ','.join(['?' for _ in sub_problem_ids])
                        cursor.execute(f"""
                            SELECT * FROM solution_attempts 
                            WHERE sub_problem_id IN ({placeholders})
                        """, sub_problem_ids)
                        solution_attempts = [dict(row) for row in cursor.fetchall()]
                        workflow_data['solution_attempts'] = solution_attempts
                
                # Write to export file
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(workflow_data, f, indent=2, default=str)
                
            logger.info(f"Workflow {workflow_id} exported to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Workflow export failed: {e}")
            return False
    
    def import_workflow(self, import_path: str) -> bool:
        """Import a workflow from a file"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            
            # Connect to target database
            with sqlite3.connect(self.target_db_path) as target_conn:
                cursor = target_conn.cursor()
                
                # Import problems
                if 'problems' in workflow_data:
                    for problem in workflow_data['problems']:
                        cursor.execute("""
                            INSERT OR REPLACE INTO problems 
                            (id, title, description, problem_type, domain_context, 
                             complexity_score, constraints, success_criteria, 
                             stakeholders, resources_available, deadline, 
                             created_at, updated_at, metadata) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            problem['id'], problem['title'], problem['description'],
                            problem['problem_type'], problem['domain_context'],
                            problem['complexity_score'], problem['constraints'],
                            problem['success_criteria'], problem['stakeholders'],
                            problem['resources_available'], problem['deadline'],
                            problem['created_at'], problem['updated_at'],
                            problem['metadata']
                        ))
                
                # Import decomposition plans
                if 'decomposition_plans' in workflow_data:
                    for plan in workflow_data['decomposition_plans']:
                        cursor.execute("""
                            INSERT OR REPLACE INTO decomposition_plans 
                            (id, problem_id, strategy, sub_problems, dependency_graph,
                             validation_checkpoints, quality_scores, confidence_level,
                             created_by, approved_by, status, created_at, updated_at, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            plan['id'], plan['problem_id'], plan['strategy'],
                            plan['sub_problems'], plan['dependency_graph'],
                            plan['validation_checkpoints'], plan['quality_scores'],
                            plan['confidence_level'], plan['created_by'],
                            plan['approved_by'], plan['status'], plan['created_at'],
                            plan['updated_at'], plan['metadata']
                        ))
                
                # Import sub-problems
                if 'sub_problems' in workflow_data:
                    for sub_problem in workflow_data['sub_problems']:
                        cursor.execute("""
                            INSERT OR REPLACE INTO sub_problems 
                            (id, parent_id, title, description, type, complexity_score,
                             dependencies, success_criteria, validation_gauntlet,
                             assigned_team, estimated_effort, priority, status,
                             created_at, updated_at, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            sub_problem['id'], sub_problem['parent_id'],
                            sub_problem['title'], sub_problem['description'],
                            sub_problem['type'], sub_problem['complexity_score'],
                            sub_problem['dependencies'], sub_problem['success_criteria'],
                            sub_problem['validation_gauntlet'], sub_problem['assigned_team'],
                            sub_problem['estimated_effort'], sub_problem['priority'],
                            sub_problem['status'], sub_problem['created_at'],
                            sub_problem['updated_at'], sub_problem['metadata']
                        ))
                
                # Import solution attempts
                if 'solution_attempts' in workflow_data:
                    for attempt in workflow_data['solution_attempts']:
                        cursor.execute("""
                            INSERT OR REPLACE INTO solution_attempts 
                            (id, sub_problem_id, approach, solution_content, team_id,
                             confidence_score, validation_results, feedback,
                             status, created_at, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            attempt['id'], attempt['sub_problem_id'],
                            attempt['approach'], attempt['solution_content'],
                            attempt['team_id'], attempt['confidence_score'],
                            attempt['validation_results'], attempt['feedback'],
                            attempt['status'], attempt['created_at'],
                            attempt['metadata']
                        ))
                
                target_conn.commit()
            
            logger.info(f"Workflow imported from {import_path}")
            return True
        except Exception as e:
            logger.error(f"Workflow import failed: {e}")
            return False


class ConfigurationManager:
    """Manages environment configuration"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            default_config = {
                "development": {
                    "debug": True,
                    "database_url": "sqlite:///sovereign_decomposition.db",
                    "log_level": "DEBUG",
                    "openevolve": {
                        "api_base": "http://localhost:8000",
                        "timeout": 30
                    },
                    "security": {
                        "encryption_key": "dev-key-change-in-production",
                        "jwt_secret": "dev-jwt-secret-change-in-production"
                    }
                },
                "staging": {
                    "debug": False,
                    "database_url": "postgresql://user:pass@db-staging:5432/sovereign",
                    "log_level": "INFO",
                    "openevolve": {
                        "api_base": "http://openevolve-staging:8000",
                        "timeout": 30
                    },
                    "security": {
                        "encryption_key": os.getenv("ENCRYPTION_KEY"),
                        "jwt_secret": os.getenv("JWT_SECRET")
                    }
                },
                "production": {
                    "debug": False,
                    "database_url": "postgresql://user:pass@db-prod:5432/sovereign",
                    "log_level": "INFO",
                    "openevolve": {
                        "api_base": "http://openevolve-prod:8000",
                        "timeout": 30
                    },
                    "security": {
                        "encryption_key": os.getenv("ENCRYPTION_KEY"),
                        "jwt_secret": os.getenv("JWT_SECRET")
                    }
                }
            }
            
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    
    def get_config_for_environment(self, environment: str = None) -> Dict[str, Any]:
        """Get configuration for specific environment"""
        env = environment or os.getenv('SOVEREIGN_ENV', 'development')
        return self.config.get(env, self.config['development'])
    
    def validate_config(self, environment: str = None) -> List[str]:
        """Validate configuration for environment"""
        config = self.get_config_for_environment(environment)
        errors = []
        
        # Check required fields
        required_fields = ['database_url', 'log_level', 'openevolve', 'security']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required configuration field: {field}")
        
        # Validate OpenEvolve config
        if 'openevolve' in config:
            openevolve = config['openevolve']
            if not openevolve.get('api_base'):
                errors.append("OpenEvolve API base URL not configured")
            if not isinstance(openevolve.get('timeout'), int):
                errors.append("OpenEvolve timeout must be an integer")
        
        # Validate security config
        if 'security' in config:
            security = config['security']
            if not security.get('jwt_secret'):
                errors.append("JWT secret not configured")
        
        return errors
    
    def update_config(self, environment: str, updates: Dict[str, Any]) -> bool:
        """Update configuration for environment"""
        try:
            self.config[environment].update(updates)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration updated for environment: {environment}")
            return True
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return False


class OperationalManager:
    """Main operational management class"""
    
    def __init__(self):
        self.deployment_manager = DeploymentManager()
        self.backup_manager = BackupManager(self.deployment_manager.config)
        self.config_manager = ConfigurationManager()
    
    def run_deployment(self, environment: str = None) -> bool:
        """Run full deployment process"""
        return self.deployment_manager.deploy(environment)
    
    def create_system_backup(self, name: str = None) -> Optional[str]:
        """Create a full system backup"""
        return self.backup_manager.create_backup(name, include_logs=True)
    
    def restore_system_backup(self, backup_path: str) -> bool:
        """Restore from backup"""
        return self.backup_manager.restore_backup(backup_path)
    
    def export_workflow(self, workflow_id: str, export_path: str) -> bool:
        """Export a workflow"""
        return self.deployment_manager.backup_current_deployment()  # Placeholder - would need actual migration manager
        
    def import_workflow(self, import_path: str) -> bool:
        """Import a workflow"""
        return True  # Placeholder - would need actual migration manager
    
    def validate_configuration(self, environment: str = None) -> bool:
        """Validate system configuration"""
        errors = self.config_manager.validate_config(environment)
        if errors:
            logger.error(f"Configuration validation errors: {errors}")
            return False
        return True
    
    def cleanup_system(self, days_to_keep: int = 30) -> int:
        """Clean up old backups and logs"""
        return self.backup_manager.cleanup_old_backups(days_to_keep)


# Global operational manager instance
_operational_manager = None


def get_operational_manager() -> OperationalManager:
    """Get the operational manager instance"""
    global _operational_manager
    if _operational_manager is None:
        _operational_manager = OperationalManager()
    return _operational_manager


def deploy_system(environment: str = None) -> bool:
    """Deploy the system"""
    return get_operational_manager().run_deployment(environment)


def backup_system(name: str = None) -> Optional[str]:
    """Create a system backup"""
    return get_operational_manager().create_system_backup(name)


def restore_system(backup_path: str) -> bool:
    """Restore system from backup"""
    return get_operational_manager().restore_system_backup(backup_path)


def validate_system_config(environment: str = None) -> bool:
    """Validate system configuration"""
    return get_operational_manager().validate_configuration(environment)


def cleanup_system_data(days_to_keep: int = 30) -> int:
    """Clean up old data"""
    return get_operational_manager().cleanup_system(days_to_keep)


def main():
    """Main command line interface for operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sovereign System Operations')
    parser.add_argument('command', choices=['deploy', 'backup', 'restore', 'validate', 'cleanup', 'export', 'import'])
    parser.add_argument('--environment', '-e', default=None, help='Target environment')
    parser.add_argument('--backup-path', '-b', help='Backup file path')
    parser.add_argument('--workflow-id', help='Workflow ID for export/import')
    parser.add_argument('--export-path', help='Export file path')
    parser.add_argument('--import-path', help='Import file path')
    parser.add_argument('--days', type=int, default=30, help='Days to keep for cleanup')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    operational_manager = get_operational_manager()
    
    if args.command == 'deploy':
        success = operational_manager.run_deployment(args.environment)
        print(f"Deployment {'successful' if success else 'failed'}")
        sys.exit(0 if success else 1)
    
    elif args.command == 'backup':
        backup_path = operational_manager.create_system_backup()
        if backup_path:
            print(f"Backup created: {backup_path}")
            sys.exit(0)
        else:
            print("Backup failed")
            sys.exit(1)
    
    elif args.command == 'restore':
        if not args.backup_path:
            print("Backup path required for restore")
            sys.exit(1)
        
        success = operational_manager.restore_system_backup(args.backup_path)
        print(f"Restore {'successful' if success else 'failed'}")
        sys.exit(0 if success else 1)
    
    elif args.command == 'validate':
        success = operational_manager.validate_configuration(args.environment)
        print(f"Configuration validation {'passed' if success else 'failed'}")
        sys.exit(0 if success else 1)
    
    elif args.command == 'cleanup':
        cleaned = operational_manager.cleanup_system(args.days)
        print(f"Cleaned up {cleaned} old files")
        sys.exit(0)
    
    elif args.command == 'export':
        if not args.workflow_id or not args.export_path:
            print("Workflow ID and export path required for export")
            sys.exit(1)
        
        # This would require a migration manager with source and target DB paths
        print("Export functionality would be implemented with migration manager")
        sys.exit(0)
    
    elif args.command == 'import':
        if not args.import_path:
            print("Import path required for import")
            sys.exit(1)
        
        print("Import functionality would be implemented with migration manager")
        sys.exit(0)


if __name__ == "__main__":
    main()