#!/usr/bin/env python3
"""
DeepKE Installation Script for OpenEvolve

This script installs DeepKE and its dependencies to enable
actual DeepKE calls (not fallback) in the knowledge extraction system.

Usage:
    python setup_deepke.py
    python setup_deepke.py --force  # Force reinstall
    python setup_deepke.py --gpu    # Install with GPU support
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
        logger.error("Python 3.8+ required for DeepKE")
        return False
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro} ✓")
    return True


def install_deepke_core(gpu=False):
    """Install DeepKE core package."""
    logger.info("Installing DeepKE...")
    
    # Base packages
    packages = [
        "deepke>=2.2.0",
        "torch>=2.0.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "seqeval>=1.2.2",
        "pytorch-crf>=0.7.2",
    ]
    
    # Add GPU support if requested
    if gpu:
        logger.info("GPU support enabled")
        # torch will use CUDA if available
    
    for package in packages:
        logger.info(f"Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", package
            ])
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")
            return False
    
    return True


def install_deepke_extras():
    """Install extra dependencies for full DeepKE functionality."""
    logger.info("Installing DeepKE extras...")
    
    extras = [
        "spacy>=3.0.0",  # For text processing
        "nltk>=3.6",     # For NLP utilities
        "scikit-learn>=1.0",  # For ML utilities
        "tqdm",          # For progress bars
        "tensorboard",   # For training visualization
    ]
    
    for package in extras:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        except subprocess.CalledProcessError as e:
            logger.warning(f"Optional package {package} failed: {e}")
    
    return True


def verify_installation():
    """Verify DeepKE is properly installed."""
    logger.info("Verifying DeepKE installation...")
    
    try:
        import deepke
        from deepke import NERModel, REModel
        
        logger.info(f"DeepKE version: {deepke.__version__ if hasattr(deepke, '__version__') else 'installed'} ✓")
        logger.info("DeepKE modules imported successfully ✓")
        
        # Check torch
        import torch
        logger.info(f"PyTorch version: {torch.__version__} ✓")
        logger.info(f"CUDA available: {torch.cuda.is_available()} ✓")
        
        # Check transformers
        import transformers
        logger.info(f"Transformers version: {transformers.__version__} ✓")
        
        return True
    except ImportError as e:
        logger.error(f"Verification failed: {e}")
        return False


def download_pretrained_models():
    """Download pretrained DeepKE models."""
    logger.info("Downloading pretrained models...")
    
    try:
        # Import after installation
        from deepke import NERModel, REModel
        import torch
        
        # Create models directory
        models_dir = Path("models/deepke")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Pretrained models will be downloaded on first use")
        logger.info(f"Models directory: {models_dir.absolute()}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to setup models: {e}")
        return False


def setup_configuration():
    """Setup DeepKE configuration files."""
    logger.info("Setting up DeepKE configuration...")
    
    config_dir = Path("config/deepke")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create default config
    config_content = """# DeepKE Configuration for OpenEvolve

# Model settings
model:
  ner_model: "deepke_ner_re"
  re_model: "deepke_ner_re"
  device: "auto"  # auto, cpu, cuda
  confidence_threshold: 0.5

# Entity types
entity_types:
  - PERSON
  - ORG
  - TECH
  - CONCEPT
  - ALGORITHM
  - METHOD
  - SYSTEM

# Relation types
relation_types:
  - USES
  - IMPLEMENTS
  - DEPENDS_ON
  - PART_OF
  - INSTANCE_OF

# Processing
processing:
  batch_size: 32
  max_length: 512
  language: "en"
"""
    
    config_file = config_dir / "deepke_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    logger.info(f"Configuration saved to: {config_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Install DeepKE for OpenEvolve")
    parser.add_argument("--force", action="store_true", help="Force reinstall")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU support")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification")
    args = parser.parse_args()
    
    print("=" * 70)
    print("DeepKE Installation for OpenEvolve")
    print("=" * 70)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install core packages
    if not install_deepke_core(gpu=args.gpu):
        logger.error("Core installation failed")
        sys.exit(1)
    
    # Install extras
    install_deepke_extras()
    
    # Setup configuration
    setup_configuration()
    
    # Download models
    download_pretrained_models()
    
    # Verify installation
    if not args.skip_verify:
        if verify_installation():
            print("\n" + "=" * 70)
            print("DeepKE Installation SUCCESSFUL ✓")
            print("=" * 70)
            print("\nDeepKE is now installed and will be used for:")
            print("  - Named Entity Recognition (NER)")
            print("  - Relation Extraction (RE)")
            print("  - Knowledge graph construction")
            print("\nRun 'python -c \"from deepke import NERModel; print('OK')\"' to verify")
        else:
            print("\n" + "=" * 70)
            print("DeepKE Installation FAILED ✗")
            print("=" * 70)
            sys.exit(1)
    
    print("\nNext steps:")
    print("  1. Run tests: pytest test_knowledge_extraction_true_100.py -v")
    print("  2. Try extraction: python integrations/deepke/bridge.py")


if __name__ == "__main__":
    main()
