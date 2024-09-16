# Email Classification using Support Vector Machine (SVM)

## Project Overview
This project aims to classify emails using different kernel types of **Support Vector Machine (SVM)**. The dataset contains various email-related features, and the target is to determine whether the emails are spam or not. We experiment with different SVM kernels: **linear**, **rbf**, **poly**, and **sigmoid** to compare their performance.

## Table of Contents
1. [Installation](#installation)
2. [Dataset Overview](#dataset-overview)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Conclusion](#conclusion)

## Installation
Before running this project, make sure to install the following dependencies:

```bash
pip install pandas numpy scikit-learn
```
## Dataset Overview
The dataset used in this project is emails.csv, which contains features about emails such as their content, sender details, etc. The target variable (last column) indicates whether an email is spam (1) or not spam (0).

Rows: 57,000+ (emails)
Columns: 300+ (features related to the emails)
Target Variable: Spam label (0 for non-spam, 1 for spam)
Example Data
Column 1	Column 2	Column 3	...	Spam
...	...	...	...	1
...	...	...	...	0
## Data Preprocessing
Import Libraries: Import the necessary libraries to manipulate data, build models, and perform evaluations.

```python

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```
Load Dataset: Load the dataset into a Pandas DataFrame and display the first 10 rows for inspection.

```python

df = pd.read_csv('/content/emails.csv')
df.head(10)
```
Data Inspection:

Check the shape of the dataset.
Look for missing values.
Inspect data types and basic information about the dataset.
```python

df.shape
df.isna().sum()
df.info()
df.dtypes
```
Drop Irrelevant Features: The column Email No. is irrelevant for our analysis, so we drop it.

```python

df = df.drop(['Email No.'], axis=1)
df.head()
```
Separate Features and Target: Split the dataset into feature variables X (all columns except the target) and the target variable Y (spam label).

```python

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
```
Normalize Features: Normalize the feature values to a range between 0 and 1 using Min-Max Scaling.

```python

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
```
Train-Test Split: Split the dataset into training and testing sets (80% training, 20% testing).

```python

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```
## Model Training and Evaluation
We train the model using four different SVM kernels: linear, rbf, poly, and sigmoid. After training each model, we calculate and compare their accuracy scores.

Linear Kernel: The linear kernel SVM typically works well for linearly separable data.

```python

model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('Accuracy:', acc)
# Accuracy: 0.9700
```
RBF Kernel: The RBF kernel (Radial Basis Function) is a commonly used non-linear kernel for SVM.

```python

model = SVC(kernel='rbf')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('Accuracy:', acc)
# Accuracy: 0.9633
```
Poly Kernel: The poly kernel is used to fit a polynomial boundary between classes, useful for complex non-linear data.

```python

model = SVC(kernel='poly')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('Accuracy:', acc)
# Accuracy: 0.7604
```
Sigmoid Kernel: The sigmoid kernel is less commonly used but can be effective in some scenarios.

```python

model = SVC(kernel='sigmoid')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('Accuracy:', acc)
# Accuracy: 0.8628
```
## Results and Comparison
- Kernel	Accuracy
- Linear	97.00%
- RBF	96.33%
- Poly	76.04%
- Sigmoid	86.28%
- Best Model: The linear kernel provides the highest accuracy (97.00%) for this dataset.
- RBF Kernel: Performs almost as well as the linear kernel.
- Poly Kernel: Struggles to fit the data, leading to a lower accuracy.
- Sigmoid Kernel: Performs reasonably well, but not as good as the linear and RBF kernels.
## Conclusion
This project demonstrates the effectiveness of the SVM algorithm for email classification using different kernel types. We found that the linear kernel produced the best accuracy, followed closely by the RBF kernel. Future work could involve tuning hyperparameters and experimenting with other algorithms to further improve performance.


