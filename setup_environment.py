# setup_environment.py

import os
import subprocess
import sys

def install_requirements(requirements_path="requirements.txt"):
    with open(requirements_path) as file:
        packages = [line.strip() for line in file if not line.startswith("#")]
    for pkg in packages:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

if __name__ == "__main__":
    print("Installing requirements...")
    install_requirements()
    print("✅ Environment setup complete.")
