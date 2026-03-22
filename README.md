# Implementation of Multivariate Linear Regression 
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
<br>import pandas as pd.

### Step2
<br>Read the csv file.

### Step3
<br>Get the value of X and y variables


### Step4
<br>Create the linear regression model and fit.

### Step5
<br>Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm cube.


## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics

# ✅ load the California housing dataset (replacement for Boston)
housing = datasets.fetch_california_housing()

# defining feature matrix (X) and response vector (y)
X = housing.data
y = housing.target

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print("Coefficients:", reg.coef_)

# variance score: 1 means perfect prediction
print("Variance score: {}".format(reg.score(X_test, y_test)))

# plot for residual error
plt.style.use("fivethirtyeight")

# plotting residual errors in training data
plt.scatter(
    reg.predict(X_train),
    reg.predict(X_train) - y_train,
    color="green",
    s=10,
    label="Train data",
)

# plotting residual errors in test data
plt.scatter(
    reg.predict(X_test),
    reg.predict(X_test) - y_test,
    color="blue",
    s=10,
    label="Test data",
)

# plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=max(reg.predict(X_test)), linewidth=2)

# plotting legend
plt.legend(loc="upper right")

# plot title
plt.title("Residual errors")

# show plot
plt.show()

```
## Output:

<img width="1453" height="847" alt="Screenshot 2026-02-14 082957" src="https://github.com/user-attachments/assets/8622e52a-71d5-458b-a463-957050d5022d" />

<img width="1394" height="767" alt="Screenshot 2026-02-14 083019" src="https://github.com/user-attachments/assets/73542f6f-e963-4019-bdbf-fcd813334ad4" />

<img width="1406" height="724" alt="Screenshot 2026-02-14 083036" src="https://github.com/user-attachments/assets/c532b719-8177-401d-b9b1-21c5f8668d7b" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
