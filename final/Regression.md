# Linear Regression and Logistic Regression


```python
# Packages
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```


```python
# Model selection and evaluation tools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
```


```python
data=pd.read_csv('BatonRouge.csv')
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1080 entries, 0 to 1079
    Data columns (total 11 columns):
    Price         1080 non-null int64
    SQFT          1080 non-null int64
    Bedrooms      1080 non-null int64
    Baths         1080 non-null int64
    Age           1080 non-null int64
    Occupancy     1080 non-null int64
    Pool          1080 non-null int64
    Style         1080 non-null int64
    Fireplace     1080 non-null int64
    Waterfront    1080 non-null int64
    DOM           1080 non-null int64
    dtypes: int64(11)
    memory usage: 92.9 KB
    


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
      <th>SQFT</th>
      <th>Bedrooms</th>
      <th>Baths</th>
      <th>Age</th>
      <th>Occupancy</th>
      <th>Pool</th>
      <th>Style</th>
      <th>Fireplace</th>
      <th>Waterfront</th>
      <th>DOM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>66500</td>
      <td>741</td>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <td>1</td>
      <td>66000</td>
      <td>741</td>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <td>2</td>
      <td>68500</td>
      <td>790</td>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <td>3</td>
      <td>102000</td>
      <td>2783</td>
      <td>2</td>
      <td>2</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <td>4</td>
      <td>54000</td>
      <td>1165</td>
      <td>2</td>
      <td>1</td>
      <td>35</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>190</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = data['Price']
x = data['SQFT']
```


```python
#model fitting find coefficient: finc beta0 and beta1
import statsmodels.api as sm


x_with_intercept = sm.add_constant(x, prepend=True)

x_with_intercept.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>SQFT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.0</td>
      <td>741</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.0</td>
      <td>741</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.0</td>
      <td>790</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.0</td>
      <td>2783</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.0</td>
      <td>1165</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = sm.OLS(y, x_with_intercept)
results = model.fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  Price   R-squared:                       0.579
    Model:                            OLS   Adj. R-squared:                  0.578
    Method:                 Least Squares   F-statistic:                     1480.
    Date:                Sun, 07 Jun 2020   Prob (F-statistic):          1.54e-204
    Time:                        11:03:43   Log-Likelihood:                -13722.
    No. Observations:                1080   AIC:                         2.745e+04
    Df Residuals:                    1078   BIC:                         2.746e+04
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const      -6.086e+04   6110.187     -9.961      0.000   -7.29e+04   -4.89e+04
    SQFT          92.7474      2.411     38.476      0.000      88.018      97.477
    ==============================================================================
    Omnibus:                     1185.147   Durbin-Watson:                   1.886
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           139602.251
    Skew:                           5.135   Prob(JB):                         0.00
    Kurtosis:                      57.743   Cond. No.                     6.38e+03
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 6.38e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    


```python
new_house = [1, 1500]
results.predict(new_house)
```




    array([78259.59937286])



# Logistic Regression

C：类似于lasso和ridge里的λ，C越大越接近于纯逻辑回归（类比与λ越小越接近纯线性回归。）如果不填默认为1.0

penalty：l1类似于lasso，l2类似于ridge。如果不填，默认为l2。

class_weight: 在imbalance classification问题中，我们希望出现较少的那个class也能被充分考虑，可以将 class_weight 设置为 'balanced'，即 class_weight='balanced'。如果不填，默认为None，也就是我们最基本的逻辑回归。


```python
from sklearn.datasets import load_breast_cancer
data_dict = load_breast_cancer()
X = data_dict.data
y = data_dict.target
```


```python
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
print(lr_model)
```

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)
    


```python
lr_model.fit(X,y) 
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)




```python
lr_model.coef_
```




    array([[ 2.14255611e+00,  1.19509857e-01, -7.76065614e-02,
            -2.63952341e-03, -1.55150939e-01, -4.12221000e-01,
            -6.55000796e-01, -3.43525103e-01, -2.27461843e-01,
            -2.68287285e-02, -2.12777314e-02,  1.29020735e+00,
             1.95855311e-02, -9.66546553e-02, -1.68391816e-02,
             1.00285617e-03, -5.13777284e-02, -4.04958496e-02,
            -4.29002319e-02,  5.92543569e-03,  1.29601147e+00,
            -3.48075791e-01, -1.20237099e-01, -2.47712838e-02,
            -2.87288179e-01, -1.17582446e+00, -1.62124781e+00,
            -6.62235441e-01, -6.99945260e-01, -1.18401588e-01]])




```python
lr_model.predict(x_test)
```
