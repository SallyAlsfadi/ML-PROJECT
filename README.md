# Machine Learning Models

This project implements core machine learning models from scratch in Python, without using any external ML libraries. The models are tested on various datasets, and the goal is to compare their performance using accuracy, precision, recall, F1 score, and confusion matrix.

## Installation

Installing pip on Windows
Step 1: Download get-pip.py via curl

If you have Python installed and curl is available (usually included in Windows 10+), run:
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

If curl is not recognized, you can try using Invoke-WebRequest in PowerShell:
Step 2: Run the script with Python
python get-pip.py

Or, if you're using the py launcher:
py get-pip.py

Step 3: Verify installation
pip --version
C:\Users\<YourUsername>\AppData\Local\Programs\Python\PythonXY\Scripts\
After installation, if pip is still not recognized, add the Python Scripts directory to your system PATH. It’s usually located at:
1. Clone the repository:

```bash
git clone https://github.com/SallyAlsfadi/ML-PROJECT.git
cd ml-project-sallyalsafadi



pip install -r requirements.txt

# Requirements
Python 3.8+
Libraries: numpy, pandas, matplotlib

# How to Run the Project
Each dataset has its own test file under tests/. To run a model on a dataset:
# Test all models on Breast Cancer dataset
python3 tests/test_all_models_breast_cancer.py or if you are using python not python3 just replace it.
 # Heart Failure dataset test
 python3 tests/test_all_models_Heart_Failure_dataset.py
  or
  python tests/test_all_models_Heart_Failure_dataset.py

# Mushroom dataset
python3 tests/test_all_models_mushroom.py

# Wine Quality dataset
python3 tests/test_all_models_wine_quality.py

# Robot Execution Failures dataset
python3 tests/test_all_models_robot_failures.py
```
