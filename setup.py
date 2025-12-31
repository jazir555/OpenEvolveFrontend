"""
OpenEvolve Frontend Package Setup

This setup.py makes the OpenEvolve frontend components properly importable
as a local package and handles all dependencies correctly.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
def read_requirements(filename):
    """Read requirements from file, skipping comments and -e flags."""
    requirements = []
    filepath = this_directory / filename
    if filepath.exists():
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Skip -e (editable) references
                    if not line.startswith('-e'):
                        requirements.append(line)
    return requirements

setup(
    name="openevolve-frontend",
    version="1.0.0",
    author="OpenEvolve Team",
    author_email="contact@openevolve.com",
    description="OpenEvolve - Sovereign-Grade Problem Decomposition and Evolution System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openevolve/openevolve",
    packages=find_packages(exclude=["tests*", "docs*", "examples*", "env*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "testing": read_requirements("requirements_with_testing.txt"),
        "hephaestus": [
            "hephaestus-client>=0.1.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.980",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "openevolve=main:main",
            "openevolve-api=api_server:main",
            "openevolve-config=config_loader:main",
        ],
    },
    include_package_data=True,
    package_data={
        "openevolve": ["*.yaml", "*.json", "templates/*.html"],
    },
    zip_safe=False,
    keywords="ai evolution decomposition optimization llm machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/openevolve/openevolve/issues",
        "Source": "https://github.com/openevolve/openevolve",
        "Documentation": "https://docs.openevolve.com",
    },
)
