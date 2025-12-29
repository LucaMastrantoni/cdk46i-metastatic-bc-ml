# Predicting Progression-Free Survival in Hormone-Receptor Positive Metastatic Breast Cancer treated with CDK4/6 Inhibitors

Repository for "**Predicting Progression-Free Survival in Hormone-Receptor Positive Metastatic Breast Cancer treated with CDK4/6 Inhibitors: A Machine Learning Approach.**". 

THEBREAST-D-25-1335

Authors: *Pannunzio S, Mastrantoni L, xxx*

Correspondence: *Mastrantoni Luca, [luca.mastrantoni01@icatt.it]*

This repository provides a step-by-step tutorial for inference and indipendent validation of the models evaluated in the manuscript xxx-

The web application is available at: xxx

## Getting Started

Follow the instructions below to clone the repository and set up the environment.

### 1. Clone the repository or download it from GitHub
Make sure you have [Git](https://git-scm.com/downloads) installed.
```bash
git clone https://github.com/LucaMastrantoni/cdk46i-metastatic-bc-ml.git
```

### 2. Create and activate the Conda environment
Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) installed.

Using Anaconda Prompt:

**a.** Go to the directory 
```bash
cd cdk46i-metastatic-bc-ml-main\Inference
```

**b.** Create conda enviroment (this may take few minutes)
```bash
conda env create -f env.yml
```

**c.** Activate the enviroment
```bash
conda activate cdk_inference
```

### 3. Run the notebook

Open the tutorial notebook Inference.ipynb. Select the "cdk_inference" enviroment.
The notebook provides step-by-step instructions for using the models to perform inference on new data.
We recommend using Visual Studio Code or PyCharm.

#### Input preparation

The data should have columns names: 

xxx should be coded as. All other variables should be numeric. 
Please refer to the example provided.

#### Notebook details

**a.** Load the required packages for model inference and visualization 
```python
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
xxx
```

**b.** Load your data (to import xlsx file use "pip install openpyxl")
```python
input_data = pd.read_excel('sample.xlsx')
```

**c.** Load the models
```python
model_cox = joblib.load("cox.sav")
model_gbm = joblib.load("gbm.sav")
```

**d.** Get predictions
```python
# Handgrip
prediction_handgrip = model_handgrip.predict(input_data, quantiles=quantiles)
quantile_handgrip_df = pd.DataFrame(prediction_handgrip, columns=[f"{q:.3f}" for q in quantiles])

handgrip_df = pd.concat([input_data, quantile_handgrip_df], axis=1)
handgrip_df
```

## Contact
For any questions, please refer to [luca.mastrantoni01@icatt.it].
