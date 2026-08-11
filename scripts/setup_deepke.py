#!/usr/bin/env python3
"""
DeepKE Installation Script for OpenEvolve - ACTUALLY WORKING VERSION

This script ACTUALLY installs DeepKE with multiple fallback methods
to ensure true 100% Knowledge Extraction functionality.

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


def install_torch(gpu=False):
    """Install PyTorch (required dependency)."""
    logger.info("Installing PyTorch...")
    
    try:
        if gpu:
            # Try CUDA 11.8 version
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
        else:
            # CPU version
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio"
            ])
        logger.info("PyTorch installed ✓")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"PyTorch installation failed: {e}")
        return False


def install_deepke_method_1():
    """Method 1: Standard pip install."""
    logger.info("Trying Method 1: Standard pip install...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--upgrade", "deepke>=2.2.0"
        ])
        logger.info("Method 1 succeeded ✓")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Method 1 failed: {e}")
        return False


def install_deepke_method_2():
    """Method 2: Install from GitHub repository."""
    logger.info("Trying Method 2: GitHub repository...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/zjunlp/DeepKE.git"
        ])
        logger.info("Method 2 succeeded ✓")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Method 2 failed: {e}")
        return False


def install_deepke_method_3():
    """Method 3: Clone and install locally."""
    logger.info("Trying Method 3: Clone and local install...")
    
    deepke_dir = Path("DeepKE_repo")
    
    try:
        # Remove existing directory
        if deepke_dir.exists():
            import shutil
            shutil.rmtree(deepke_dir)
        
        # Clone repository
        subprocess.check_call([
            "git", "clone",
            "https://github.com/zjunlp/DeepKE.git",
            str(deepke_dir)
        ])
        
        # Install from source
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-e", str(deepke_dir)
        ])
        
        logger.info("Method 3 succeeded ✓")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Method 3 failed: {e}")
        return False


def install_deepke_method_4():
    """Method 4: Install with --no-deps and manually install dependencies."""
    logger.info("Trying Method 4: Manual dependency resolution...")
    
    try:
        # Install core dependencies first
        deps = [
            "transformers>=4.20.0",
            "datasets>=2.0.0",
            "seqeval>=1.2.2",
            "pytorch-crf>=0.7.2",
            "tqdm",
            "numpy",
        ]
        
        for dep in deps:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", dep
                ])
            except:
                pass
        
        # Try to install deepke without deps
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--no-deps", "deepke"
        ])
        
        logger.info("Method 4 succeeded ✓")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Method 4 failed: {e}")
        return False


def install_deepke_core(gpu=False):
    """Install DeepKE core package with multiple methods."""
    logger.info("Installing DeepKE (trying multiple methods)...")
    
    # First ensure torch is installed
    try:
        import torch
        logger.info(f"PyTorch already installed: {torch.__version__}")
    except ImportError:
        if not install_torch(gpu=gpu):
            logger.error("Failed to install PyTorch")
            return False
    
    # Try multiple installation methods
    methods = [
        install_deepke_method_1,
        install_deepke_method_2,
        install_deepke_method_3,
        install_deepke_method_4,
    ]
    
    for method in methods:
        if method():
            # Verify it actually works
            if verify_installation_quiet():
                return True
            else:
                logger.warning(f"Method reported success but verification failed, trying next...")
    
    logger.error("All installation methods failed")
    return False


def install_deepke_extras():
    """Install extra dependencies for full DeepKE functionality."""
    logger.info("Installing DeepKE extras...")
    
    extras = [
        "spacy>=3.0.0",
        "nltk>=3.6",
        "scikit-learn>=1.0",
        "tensorboard",
    ]
    
    for package in extras:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        except subprocess.CalledProcessError:
            logger.warning(f"Optional package {package} failed to install")
    
    return True


def verify_installation_quiet():
    """Quick verification without logging."""
    try:
        import deepke
        from deepke import NERModel, REModel
        import torch
        return True
    except ImportError:
        return False


def verify_installation():
    """Verify DeepKE is properly installed and can actually be used."""
    logger.info("=" * 70)
    logger.info("VERIFYING DeepKE Installation")
    logger.info("=" * 70)
    
    success = True
    
    # Check 1: Import
    try:
        import deepke
        logger.info("✓ DeepKE module imports successfully")
    except ImportError as e:
        logger.error(f"✗ DeepKE import failed: {e}")
        success = False
    
    # Check 2: Import NERModel
    try:
        from deepke import NERModel
        logger.info("✓ NERModel imports successfully")
    except ImportError as e:
        logger.error(f"✗ NERModel import failed: {e}")
        success = False
    
    # Check 3: Import REModel
    try:
        from deepke import REModel
        logger.info("✓ REModel imports successfully")
    except ImportError as e:
        logger.error(f"✗ REModel import failed: {e}")
        success = False
    
    # Check 4: PyTorch
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__} installed")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        logger.error(f"✗ PyTorch import failed: {e}")
        success = False
    
    # Check 5: Transformers
    try:
        import transformers
        logger.info(f"✓ Transformers {transformers.__version__} installed")
    except ImportError as e:
        logger.error(f"✗ Transformers import failed: {e}")
        success = False
    
    logger.info("=" * 70)
    if success:
        logger.info("DeepKE Installation VERIFIED ✓")
    else:
        logger.error("DeepKE Installation INCOMPLETE ✗")
    logger.info("=" * 70)
    
    return success


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
    parser.add_argument("--verify-only", action="store_true", help="Only verify installation")
    args = parser.parse_args()
    
    print("=" * 70)
    print("DeepKE Installation for OpenEvolve - TRUE 100% VERSION")
    print("=" * 70)
    
    if args.verify_only:
        verify_installation()
        return
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if already installed
    if not args.force and verify_installation_quiet():
        print("\nDeepKE is already installed. Use --force to reinstall.")
        verify_installation()
        return
    
    # Install core packages
    if not install_deepke_core(gpu=args.gpu):
        logger.error("Core installation failed")
        print("\n" + "=" * 70)
        print("DeepKE Installation FAILED ✗")
        print("=" * 70)
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Try: pip install --upgrade pip setuptools wheel")
        print("3. Install Visual C++ Build Tools (Windows)")
        print("4. Try manual installation: git clone https://github.com/zjunlp/DeepKE.git")
        sys.exit(1)
    
    # Install extras
    install_deepke_extras()
    
    # Setup configuration
    setup_configuration()
    
    # Verify installation
    if not args.skip_verify:
        if verify_installation():
            print("\n" + "=" * 70)
            print("DeepKE Installation SUCCESSFUL ✓")
            print("=" * 70)
            print("\nDeepKE is now ACTUALLY installed and will be used for:")
            print("  - Named Entity Recognition (NER)")
            print("  - Relation Extraction (RE)")
            print("  - Knowledge graph construction")
            print("\nNO MORE FALLBACKS - DeepKE will be called directly!")
            print("\nNext steps:")
            print("  1. Run: python verify_knowledge_extraction.py")
            print("  2. Run: pytest test_knowledge_extraction_true_100.py -v")
        else:
            print("\n" + "=" * 70)
            print("DeepKE Installation INCOMPLETE ✗")
            print("=" * 70)
            sys.exit(1)


if __name__ == "__main__":
    main()
