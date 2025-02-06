import os
import subprocess
import shutil

def install_packages():
    subprocess.check_call(["pip", "install", "kaggle"])
    subprocess.check_call(["pip", "install", "tensorflow"])
    subprocess.check_call(["pip", "install", "keras"])

def setup_kaggle_credentials():
    kaggle_json_path = "kaggle.json"
    
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError("kaggle.json file not found. Please upload it to the current directory.")
    
    kaggle_dir = os.path.expanduser("~/.kaggle")
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
    
    shutil.copy(kaggle_json_path, kaggle_dir)
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

def download_dataset(download_folder):
    os.makedirs(download_folder, exist_ok=True)
    subprocess.check_call(["kaggle", "competitions", "download", "-c", "dogs-vs-cats", "-p", download_folder])

def unzip_dataset(download_folder):
    raw_data_folder = os.path.join(download_folder, "raw")
    os.makedirs(raw_data_folder, exist_ok=True)
    subprocess.check_call(["unzip", os.path.join(download_folder, "dogs-vs-cats.zip"), "-d", raw_data_folder])
    
    # Unzip train.zip directly into the raw directory
    subprocess.check_call(["unzip", os.path.join(raw_data_folder, "train.zip"), "-d", raw_data_folder])
    
    # Rename test1.zip to test.zip before unzipping
    os.rename(os.path.join(raw_data_folder, "test1.zip"), os.path.join(raw_data_folder, "test.zip"))
    
    # Unzip test.zip directly into the raw directory
    subprocess.check_call(["unzip", os.path.join(raw_data_folder, "test.zip"), "-d", raw_data_folder])
    
    # Rename the test1 folder to test
    test1_folder_path = os.path.join(raw_data_folder, "test1")
    test_folder_path = os.path.join(raw_data_folder, "test")
    if os.path.exists(test1_folder_path):
        os.rename(test1_folder_path, test_folder_path)

def create_gitignore():
    gitignore_content = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Jupyter Notebook
.ipynb_checkpoints

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Kaggle API credentials
kaggle.json

# dotenv environment variables file
.env

# VS Code settings
.vscode/
"""
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)

def main():
    download_folder = "data"
    
    install_packages()
    setup_kaggle_credentials()
    download_dataset(download_folder)
    unzip_dataset(download_folder)
    create_gitignore()
    print("Dataset downloaded, unzipped, and .gitignore file created successfully.")

if __name__ == "__main__":
    main()