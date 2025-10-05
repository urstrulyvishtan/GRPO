#!/usr/bin/env python3
"""
Setup script for GRPO project.

This script helps set up the GRPO project environment and dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False
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

def setup_logging():
    """Setup logging configuration"""
    print("Setting up logging...")
    log_config = """
import logging
import os

def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/grpo.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
"""
    
    with open("logging_config.py", "w") as f:
        f.write(log_config)
    
    print("✓ Logging configuration created")

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
    
    print("✓ All packages verified successfully")
    return True

def main():
    """Main setup function"""
    print("GRPO Project Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("Setup failed: Could not install requirements")
        return False
    
    # Create directories
    create_directories()
    
    # Setup logging
    setup_logging()
    
    # Verify installation
    if not verify_installation():
        print("Setup failed: Package verification failed")
        return False
    
    print("\n" + "=" * 50)
    print("✓ GRPO project setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python main.py demo' for a quick demonstration")
    print("2. Run 'python main.py train --episodes 100' to train the agent")
    print("3. Run 'python main.py pipeline' for full training and evaluation")
    print("4. Check the README.md for detailed usage instructions")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
