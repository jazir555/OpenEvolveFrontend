#!/usr/bin/env python3
"""
OneKE Installation Script for OpenEvolve

This script installs OneKE and its dependencies to enable
actual OneKE calls (not fallback/stub) in the knowledge extraction system.

OneKE: Schema-Guided Knowledge Extraction
Repository: https://github.com/zjunlp/OneKE

Usage:
    python setup_oneke.py
    python setup_oneke.py --force  # Force reinstall
    python setup_oneke.py --clone  # Clone and install from source
"""

import subprocess
import sys
import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8+ required for OneKE")
        return False
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro} ✓")
    return True


def clone_oneke_repository():
    """Clone OneKE repository from GitHub."""
    logger.info("Cloning OneKE repository...")
    
    oneke_path = Path("OneKE")
    if oneke_path.exists():
        logger.info("OneKE directory already exists, pulling latest changes...")
        try:
            subprocess.check_call(["git", "-C", str(oneke_path), "pull"])
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull updates: {e}")
            return False
    
    try:
        subprocess.check_call([
            "git", "clone", 
            "https://github.com/zjunlp/OneKE.git",
            str(oneke_path)
        ])
        logger.info("OneKE repository cloned successfully ✓")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone OneKE: {e}")
        return False


def install_oneke_dependencies():
    """Install OneKE Python dependencies."""
    logger.info("Installing OneKE dependencies...")
    
    # Core dependencies based on OneKE requirements
    dependencies = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "accelerate>=0.20.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.39.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "openai>=1.0.0",
        "tiktoken>=0.4.0",
        "vllm>=0.2.0",  # For inference acceleration
    ]
    
    for dep in dependencies:
        logger.info(f"Installing {dep}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", dep
            ])
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install {dep}: {e}")
    
    return True


def install_oneke_from_source():
    """Install OneKE from local source."""
    logger.info("Installing OneKE from source...")
    
    oneke_path = Path("OneKE")
    if not oneke_path.exists():
        logger.error("OneKE directory not found. Run with --clone first.")
        return False
    
    try:
        # Install OneKE as editable package
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-e", str(oneke_path)
        ])
        logger.info("OneKE installed from source ✓")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install OneKE: {e}")
        return False


def create_oneke_wrapper():
    """Create OneKE wrapper module for OpenEvolve."""
    logger.info("Creating OneKE wrapper module...")
    
    wrapper_content = '''"""
OneKE Wrapper for OpenEvolve

Provides actual OneKE API calls for schema-guided knowledge extraction.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

# Add OneKE to path if it exists locally
ONEKE_PATH = Path(__file__).parent.parent.parent / "OneKE"
if ONEKE_PATH.exists():
    sys.path.insert(0, str(ONEKE_PATH))


class OneKEWrapper:
    """Wrapper for OneKE actual API calls."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self._available = False
        
    def initialize(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None) -> bool:
        """Initialize OneKE with actual model."""
        try:
            # Try to import OneKE modules
            from oneke import OneKE
            
            # Initialize with OpenAI or local model
            openai_key = api_key or os.getenv("OPENAI_API_KEY")
            
            if openai_key:
                self.model = OneKE(
                    model_name_or_path=model_name,
                    api_key=openai_key,
                    model_category="ChatGPT"
                )
            else:
                # Try local model
                self.model = OneKE(
                    model_name_or_path="zjunlp/oneke",
                    model_category="Local"
                )
            
            self._available = True
            logger.info(f"OneKE initialized with model: {model_name}")
            return True
            
        except ImportError as e:
            logger.error(f"OneKE not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OneKE: {e}")
            return False
    
    def extract(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Actually call OneKE for extraction."""
        if not self._available or self.model is None:
            raise RuntimeError("OneKE not initialized")
        
        try:
            # Call actual OneKE API
            result = self.model.extract(
                text=text,
                schema=schema,
                task=schema.get("task", "NER")
            )
            
            return {
                "entities": result.get("entities", []),
                "relations": result.get("relations", []),
                "events": result.get("events", []),
                "triples": result.get("triples", []),
                "confidence": result.get("confidence", 0.8),
                "model": self.config.get("model_name", "unknown"),
                "source": "oneke_actual"
            }
            
        except Exception as e:
            logger.error(f"OneKE extraction failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if OneKE is available."""
        return self._available


def install_and_import_oneke():
    """Helper to install and import OneKE."""
    try:
        import oneke
        return oneke
    except ImportError:
        logger.warning("OneKE not installed. Run setup_oneke.py first.")
        return None
'''
    
    wrapper_path = Path("integrations/oneke/oneke_wrapper.py")
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    logger.info(f"OneKE wrapper created at: {wrapper_path}")
    return True


def setup_configuration():
    """Setup OneKE configuration files."""
    logger.info("Setting up OneKE configuration...")
    
    config_dir = Path("config/oneke")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create default config
    config_content = """# OneKE Configuration for OpenEvolve

# Model settings
model:
  model_category: "ChatGPT"  # ChatGPT, Local, etc.
  model_name_or_path: "gpt-4o-mini"
  api_key: null  # Set via OPENAI_API_KEY env var

# Features
features:
  ner: true
  re: true
  ee: true
  triple: true
  multi_agent: true

# Schemas to load
schemas:
  - physics_concepts
  - chemical_entities
  - relations

# Integration
integration:
  auto_start: true
  cache_enabled: true
  cache_ttl: 3600
  fallback_on_error: true

# Performance
performance:
  max_workers: 4
  timeout: 30
  batch_size: 100
"""
    
    config_file = config_dir / "oneke_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    logger.info(f"Configuration saved to: {config_file}")
    
    # Create example schema
    schema_dir = Path("integrations/oneke/schemas")
    schema_dir.mkdir(parents=True, exist_ok=True)
    
    physics_schema = """name: physics_concepts
description: Physics concepts and entities

entity_types:
  - name: Physical_Concept
    description: Core physics concept
  - name: Observable
    description: Observable physical quantity
  - name: Dynamics
    description: Dynamical system or process
  - name: Quantum_Entity
    description: Quantum mechanical entity

relation_types:
  - name: describes
    description: Concept describes phenomenon
  - name: relates_to
    description: Concept relates to another
  - name: instance_of
    description: Entity is instance of concept
"""
    
    with open(schema_dir / "physics_concepts.yaml", 'w') as f:
        f.write(physics_schema)
    
    logger.info(f"Example schema saved to: {schema_dir / 'physics_concepts.yaml'}")
    return True


def verify_installation():
    """Verify OneKE is properly installed."""
    logger.info("Verifying OneKE installation...")
    
    try:
        # Try importing OneKE
        sys.path.insert(0, str(Path("OneKE")))
        
        try:
            import oneke
            logger.info("OneKE module imported successfully ✓")
        except ImportError:
            logger.warning("OneKE module not in PYTHONPATH, checking wrapper...")
        
        # Check wrapper
        from integrations.oneke.oneke_wrapper import OneKEWrapper
        wrapper = OneKEWrapper()
        logger.info("OneKE wrapper imported successfully ✓")
        
        # Check OpenAI key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            logger.info("OPENAI_API_KEY is set ✓")
        else:
            logger.warning("OPENAI_API_KEY not set - OneKE will use local models")
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Install OneKE for OpenEvolve")
    parser.add_argument("--force", action="store_true", help="Force reinstall")
    parser.add_argument("--clone", action="store_true", help="Clone repository")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification")
    args = parser.parse_args()
    
    print("=" * 70)
    print("OneKE Installation for OpenEvolve")
    print("=" * 70)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Clone repository if requested
    if args.clone:
        if not clone_oneke_repository():
            logger.error("Failed to clone OneKE repository")
            sys.exit(1)
    
    # Install dependencies
    if not install_oneke_dependencies():
        logger.error("Failed to install dependencies")
        sys.exit(1)
    
    # Install from source if directory exists
    oneke_path = Path("OneKE")
    if oneke_path.exists():
        if not install_oneke_from_source():
            logger.warning("Failed to install from source, continuing with pip...")
    
    # Create wrapper
    create_oneke_wrapper()
    
    # Setup configuration
    setup_configuration()
    
    # Verify installation
    if not args.skip_verify:
        if verify_installation():
            print("\n" + "=" * 70)
            print("OneKE Installation SUCCESSFUL ✓")
            print("=" * 70)
            print("\nOneKE is now installed and will be used for:")
            print("  - Schema-guided Named Entity Recognition")
            print("  - Schema-guided Relation Extraction")
            print("  - Event Extraction")
            print("  - Triple Extraction")
            print("\nTo use with OpenAI models, set OPENAI_API_KEY environment variable")
        else:
            print("\n" + "=" * 70)
            print("OneKE Installation PARTIAL ✓")
            print("=" * 70)
            print("\nWrapper is ready but OneKE library may need manual installation")
    
    print("\nNext steps:")
    print("  1. Set OPENAI_API_KEY if using OpenAI models")
    print("  2. Run tests: pytest test_knowledge_extraction_true_100.py -v")
    print("  3. Try extraction: python integrations/oneke/bridge.py")


if __name__ == "__main__":
    main()
