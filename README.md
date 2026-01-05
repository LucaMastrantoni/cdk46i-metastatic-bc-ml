# Predicting Progression-Free Survival in Hormone-Receptor Positive Metastatic Breast Cancer treated with CDK4/6 Inhibitors

Repository for "**Predicting Progression-Free Survival in Hormone-Receptor Positive Metastatic Breast Cancer treated with CDK4/6 Inhibitors: A Machine Learning Approach**". 

THEBREAST-D-25-1335

Authors: *Pannunzio S, Mastrantoni L, xxx*

Correspondence: *Mastrantoni Luca, [luca.mastrantoni01@icatt.it]*

This repository provides a step-by-step tutorial for inference and indipendent validation of the models evaluated in the manuscript xxx-

## Getting Started

Follow the instructions below to clone the repository and set up the environment.

### 1. Clone the repository or download it from GitHub
Make sure you have [Git](https://git-scm.com/downloads) installed. Then, from command prompt, clone the repository:
```bash
git clone https://github.com/LucaMastrantoni/cdk46i-metastatic-bc-ml.git
```

### 2. Create and activate the Conda environment
Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) installed.

Using Anaconda Prompt:

**a.** Go to the directory 
```bash
cd cdk46i-metastatic-bc-ml\env
```

**b.** Create conda enviroment (this may take few minutes)
```bash
conda env create -f env.yml
```

**c.** Activate the enviroment 
```bash
conda activate cdk46i-metastatic-bc-ml
```
Check that everything is okay: (base) should change to (cdk46i-metastatic-bc-ml). Now close the Anaconda Prompt.

**c.** Open the "Validation" notebook
Set the directory in the main folder and open a notebook in your IDE. Choose the environment "cdk46i-metastatic-bc-ml" and run the code to reproduce the results of our validation cohort.

### 3. Prepare your data
The **"Datasets"** folder contains the following files:
- **"validation_dataset.xlsx"**: Excel file containing the data to replicate the validation step of the original manuscript. This SHOULD NOT be modified.
- **"Inference_dataset.xlsx"**: An Excel template containing 5 patients to be used for reference. Please prepare your dataset (or modify this file) according to this template. See "Input preparation" sections for further details.
- **"Inference_with_predictions.xlsx"**: An example on how the file should look after running the inference notebook.  

### 4. Run the notebooks
The **"Notebooks"** folder contains the following Jupyter notebooks:
- **"Validation.ipynb"**: The notebook provides step-by-step instructions to replicate the validation results of the original manuscript. The input can be modified to perform indipendent external validation and calcualate metrics on new data.
- **"Inference.ipynb"**: A ready-to-use notebook to perform inference and calculate risk scores for further analysis. 

We suggest to use the provided conda environment to run the notebooks, this can be done selecting the "cdk46i-metastatic-bc-ml" enviroment from your IDE. We recommend using Visual Studio Code or PyCharm.

## Input preparation

The data should have the following columns names and levels: 
- **"Age"**: numeric
- "

Binary variables should be coded as 1/0. All other variables should be numeric. 
The pipeline for evaluation includes a step for imputing missing data from the training dataset (to avoid information leakage). Any other methods can be used for indipendent data, as long as the imputed set is passed in the correct format. 
Please refer to the "Datasets" folder for any doubts.

## Notebook overview
This part should serve as an overview of the basic steps of the inference pipeline. Please refer to the notebook "Inference.ipynb" to perform the analysis.

**a.** Import the required packages for inference and visualization 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.datasets import get_x_y
import joblib
```

**b.** Load your data and split them into variables (X) and outcomes (y)
```python
data = pd.read_excel(data_path)
X, y = get_x_y(data, attr_labels, survival=True, pos_label=1)
```

**c.** Load the model
```python
model_gbm = joblib.load(model_path)
```

**d.** Get predictions
```python
# Predict survival functions and risk scores
surv_fns = model_gbm.predict_survival_function(X) # Predict survival function
risk_scores = model_gbm.predict(X)  # Predict risk scores (Higher risk = worse PFS)
```
**e.** Append predictions to the original dataframe.

## Contact
For any questions, please refer to [luca.mastrantoni01@icatt.it].
