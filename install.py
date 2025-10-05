#!/usr/bin/env python3
"""
Installation script for GRPO project.

This script handles the installation of all dependencies including PyTorch.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_pytorch():
    """Install PyTorch with appropriate backend"""
    print("Installing PyTorch...")
    
    # Check if CUDA is available
    try:
        import torch
        print("✓ PyTorch already installed")
        return True
    except ImportError:
        pass
    
    # Install PyTorch CPU version (works on all systems)
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        print("✓ PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install PyTorch: {e}")
        return False

def install_other_requirements():
    """Install other requirements"""
    print("Installing other requirements...")
    
    requirements = [
        "numpy>=1.21.0",
        "matplotlib>=3.5.0", 
        "seaborn>=0.11.0",
        "tqdm>=4.64.0",
        "gymnasium>=0.28.0",
        "scipy>=1.9.0",
        "pandas>=1.4.0"
    ]
    
    try:
        for req in requirements:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        print("✓ Other requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def install_optional_requirements():
    """Install optional requirements"""
    print("Installing optional requirements...")
    
    optional_requirements = [
        "tensorboard>=2.10.0",
        "plotly>=5.10.0"
    ]
    
    try:
        for req in optional_requirements:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        print("✓ Optional requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠ Optional requirements installation failed: {e}")
        print("  This is not critical - continuing...")
        return True  # Don't fail for optional requirements

def verify_installation():
    """Verify that all packages are installed correctly"""
    print("Verifying installation...")
    
    required_packages = [
        "torch",
        "numpy", 
        "matplotlib",
        "seaborn",
        "tqdm",
        "gymnasium",
        "pandas",
        "scipy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        return False
    
    print("✓ All required packages verified successfully")
    return True

def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    directories = [
        "logs",
        "models", 
        "plots",
        "results",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def test_grpo_imports():
    """Test that GRPO modules can be imported"""
    print("Testing GRPO imports...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from config import GRPOConfig
        from grpo_agent import GRPOAgent
        from uncertain_env import create_uncertain_env
        print("✓ GRPO modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import GRPO modules: {e}")
        return False

def main():
    """Main installation function"""
    print("GRPO Installation Script")
    print("=" * 50)
    
    # Install PyTorch first
    if not install_pytorch():
        print("Installation failed: Could not install PyTorch")
        return False
    
    # Install other requirements
    if not install_other_requirements():
        print("Installation failed: Could not install other requirements")
        return False
    
    # Install optional requirements
    install_optional_requirements()
    
    # Create directories
    create_directories()
    
    # Verify installation
    if not verify_installation():
        print("Installation failed: Package verification failed")
        return False
    
    # Test GRPO imports
    if not test_grpo_imports():
        print("Installation failed: GRPO modules could not be imported")
        return False
    
    print("\n" + "=" * 50)
    print("✓ GRPO installation completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python test.py' to verify everything works")
    print("2. Run 'python main.py demo' for a quick demonstration")
    print("3. Run 'python main.py train --episodes 100' to train the agent")
    print("4. Check the README.md for detailed usage instructions")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
