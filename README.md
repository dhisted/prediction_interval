
Based off the work done here:
- CQR: https://arxiv.org/abs/1905.03222
- CQR: poster concise explanation: https://github.com/yromano/cqr/blob/master/poster/CQR_Poster.pdf

### macOS users install `libomp` before running the project:
```bash
brew install libomp


```python
%pip install prediction_interval
```


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from prediction_interval.models import XGBoostCQR, XGBoostQuantileRegressor, XGBoostBootstrap
```

# Demo of how to use package:

#### Creating dummy data


```python
np.random.seed(42)
n_samples = 1000

# Generate some random features
X1 = np.random.normal(loc=0, scale=1, size=n_samples)
X2 = np.random.normal(loc=5, scale=2, size=n_samples)
X3 = np.random.normal(loc=5, scale=2, size=n_samples)
X4 = np.random.normal(loc=5, scale=2, size=n_samples)
X5 = np.random.normal(loc=5, scale=2, size=n_samples)
X6 = np.random.normal(loc=5, scale=2, size=n_samples)
X7 = np.random.normal(loc=5, scale=2, size=n_samples)

# Create a target variable with some noise
# assume a linear relationship with some noise
y = 3 * X1 + 2 * X2 + np.random.normal(loc=0, scale=1, size=n_samples) +X3+X4+X5+X6+X7

df = pd.DataFrame({
    "Feature1": X1,
    "Feature2": X2,
    "Feature3": X3,
    "Feature4": X4,
    "Feature5": X5,
    "Feature6": X6,
    "Feature7": X7,
    "Target": y
})

print(df.head())
print(df.shape)
```

       Feature1  Feature2  Feature3  Feature4  Feature5  Feature6  Feature7  \
    0  0.496714  7.798711  3.649643  1.184385  3.273013  4.152481  2.771837   
    1 -0.138264  6.849267  4.710963  3.279230  4.937593  4.093172  3.738138   
    2  0.647689  5.119261  3.415160  4.172789  5.036034  1.408714  3.115880   
    3  1.523030  3.706126  4.384077  8.775375  5.945261  4.339820  3.904008   
    4 -0.234153  6.396447  1.212771  6.113106  2.266283  6.465658  4.571699   
    
          Target  
    0  32.904108  
    1  32.265157  
    2  30.044909  
    3  39.096159  
    4  33.427409  
    (1000, 8)


##### To import data saved locally uncomment the following and add full path to dataset in place of path_to_file...


```python
# # Importing data saved locally:
# path = r"path_to_file..."
# data = pd.read_csv(path)

```

### Splitting data


```python
X_data = df.copy()
y_data = X_data.pop("Target")
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25)
X_calibration, X_test, y_calibration, y_test = train_test_split(X_test, y_test, test_size=0.5)
```


```python
xgboost_params = {
    "max_depth": 4,
    "eta": 0.05,
    "gamma": 0.1,
    "subsample": 0.8,
    "min_child_weight": 100,
    "colsample_bytree": 1,
    "alpha": 5,
    "lambda": 0,
    "base_score": 0.57322098
}
cqr_xgboost = XGBoostCQR(
    model_params=xgboost_params,
    num_boost_round=int(100),
    early_stopping_rounds=10,
    alpha=0.9  # coverage probability for the conformal prediction part of CQR - should be set to your final desired coverage level
    )
```

##### Using grid search to test different coverage probabilities for the quantile regression portion of CQR
###### Grid search is NOT a required step - can skip straight to fitting final PI model
###### The conformal prediction alpha ensures the PIs are padded to maintain the desired coverage probability even with a lower coverage specified for the quantile regression portion


```python
lower_qr_quantile, upper_qr_quantile, conformity_score = cqr_xgboost.cqr_grid_search_alpha(
    [0.1, 0.2, 0.4,],            # lower quantile coverage probability levels
    [0.6, 0.8, 0.9],             # upper quantile coverage probability levels
    X_train.values,         
    y_train.values, 
    X_calibration.values,   # unseen holdout calibration set used for the conformal prediction portion
    y_calibration.values, 
    X_test.values, 
    y_test.values, 
    0.25                    # split size for the training data of the initial quantile regression
    )
```

    Evaluating --- Lower alpha: 0.1 --- Upper alpha 0.6
    Evaluating --- Lower alpha: 0.1 --- Upper alpha 0.8
    Evaluating --- Lower alpha: 0.1 --- Upper alpha 0.9
    Evaluating --- Lower alpha: 0.2 --- Upper alpha 0.6
    Evaluating --- Lower alpha: 0.2 --- Upper alpha 0.8
    Evaluating --- Lower alpha: 0.2 --- Upper alpha 0.9
    Evaluating --- Lower alpha: 0.4 --- Upper alpha 0.6
    Evaluating --- Lower alpha: 0.4 --- Upper alpha 0.8
    Evaluating --- Lower alpha: 0.4 --- Upper alpha 0.9
    best CWC: 2.2498335501828367e-12
    best coverage: 0.952
    QR Alphas: [0.4, 0.6]
       l_alpha  u_alpha           cwc  coverage  average_PI_width
    0      0.1      0.6  3.123635e-12     0.960         19.704405
    1      0.1      0.8  4.587595e-12     0.952         16.637457
    2      0.1      0.9  7.390271e-12     0.936         16.710157
    3      0.2      0.6  4.482768e-12     0.952         17.158796
    4      0.2      0.8  3.692294e-12     0.960         16.109138
    5      0.2      0.9  7.937115e-12     0.936         15.027287
    6      0.4      0.6  2.249834e-12     0.976         16.465689
    7      0.4      0.8  3.815250e-12     0.960         15.331766
    8      0.4      0.9  4.629115e-12     0.952         16.430967


##### The output identifies the best quantile values for the quantile regression portion of the prediction interval (0.4 and 0.6)
- despite the quantile levels giving a prediction interval width of 0.6-0.4 = 0.2 the overall PI will still have a width of 0.9 due to the alpha defined at the start
##### Using the results of the grid search we can fit the CQR model with the optimum quantile levels


```python
cqr_xgboost.fit(
    X_train.values, 
    y_train.values, 
    X_calibration.values, 
    y_calibration.values, 
    0.25,                   # training/validation data split proportion used in the quantile regression portion 
    0.4,                    # lower quantile value for quantile regression portion
    0.6,                    # lower quantile value for quantile regression portion
    )
```

    Evaluating --- Lower QR quantile: 0.4 --- Upper QR quantile 0.6 --- CQR alpha 0.9



```python
cqr_preds = cqr_xgboost.predict(X_test)
cqr_preds
```

    {'model_40': <xgboost.core.Booster object at 0x000001A4BE092630>, 'model_60': <xgboost.core.Booster object at 0x000001A4BE091F40>}
    (np.float64(6.691801754053905), np.float64(6.691801754053905))





    {'model_40_predictions': array([23.486837, 18.67393 , 32.484924, 27.112062, 28.803766, 27.276163,
            22.676764, 31.016535, 33.065357, 29.88289 , 29.197748, 23.760366,
            25.238892, 28.017939, 25.820185, 30.838179, 26.705332, 25.86778 ,
            29.041285, 26.21456 , 30.407969, 27.181593, 29.95053 , 23.81866 ,
            30.059622, 29.288862, 28.555567, 28.5591  , 28.319204, 21.84705 ,
            24.585625, 26.863321, 25.627268, 24.067434, 26.742422, 26.236872,
            33.069336, 24.5562  , 30.418066, 27.926493, 21.664396, 23.109285,
            28.026175, 31.024427, 21.150793, 22.963375, 28.087214, 27.088976,
            24.280598, 22.03742 , 22.731644, 31.532808, 22.613436, 20.83436 ,
            29.681917, 29.690145, 28.256884, 30.230074, 31.142015, 29.283438,
            26.716932, 25.042675, 23.072714, 19.717453, 24.86407 , 29.446177,
            31.87668 , 26.152586, 29.279047, 29.858416, 26.830156, 25.162756,
            27.63354 , 19.20571 , 24.69247 , 25.690577, 30.029097, 30.762701,
            18.25805 , 27.989725, 24.016665, 26.895353, 30.669115, 29.7009  ,
            30.946955, 32.54754 , 24.912683, 27.601076, 31.12282 , 29.557177,
            24.888874, 25.8893  , 30.666895, 28.617975, 21.349886, 23.419872,
            28.527033, 24.727304, 30.481913, 27.833437, 22.214617, 30.06967 ,
            32.49518 , 26.203722, 23.8881  , 25.841269, 24.499422, 29.215921,
            29.807978, 22.654224, 26.073137, 28.352068, 27.04648 , 30.973734,
            31.945948, 26.06415 , 27.427275, 24.810951, 24.995848, 28.251188,
            30.708555, 18.834095, 25.407259, 23.607534, 25.937593],
           dtype=float32),
     'model_60_predictions': array([38.722267, 33.717125, 46.950207, 42.872467, 44.487522, 43.517685,
            36.2037  , 47.33104 , 49.405422, 45.57175 , 45.849594, 40.34343 ,
            42.515392, 45.768757, 41.283974, 47.002872, 43.49001 , 40.70927 ,
            44.985947, 42.033783, 45.824375, 43.750404, 45.824776, 38.606483,
            44.9471  , 44.982548, 43.323177, 44.30441 , 46.64857 , 36.690807,
            42.08947 , 43.14702 , 40.105347, 42.03816 , 42.343517, 42.82062 ,
            47.231827, 39.99295 , 46.01658 , 44.035904, 40.032692, 39.847237,
            43.620773, 45.87897 , 38.059414, 39.671467, 44.383564, 43.294315,
            42.21498 , 37.707195, 39.172314, 48.78219 , 40.33695 , 35.455032,
            45.304146, 44.3602  , 44.47025 , 44.866173, 47.033566, 45.58766 ,
            43.899357, 39.882862, 38.573433, 36.740124, 42.331272, 44.73848 ,
            48.787716, 42.771965, 43.78315 , 46.3867  , 43.134953, 38.920753,
            42.35099 , 34.952   , 40.73547 , 41.644997, 45.818905, 48.361866,
            34.787315, 42.48722 , 39.872192, 43.96925 , 46.86797 , 45.07383 ,
            48.388138, 47.090015, 40.41863 , 44.758327, 45.3852  , 44.541466,
            42.1466  , 43.65617 , 47.54282 , 44.77244 , 36.969772, 41.54296 ,
            43.831978, 39.787712, 46.23097 , 44.46332 , 39.139694, 44.293724,
            49.27917 , 42.516434, 39.81532 , 42.348747, 41.600716, 44.78814 ,
            43.9265  , 38.41995 , 42.405285, 42.678352, 40.930042, 47.90765 ,
            45.33097 , 41.709576, 40.599304, 40.70203 , 40.479744, 41.99648 ,
            48.51149 , 36.695763, 41.383724, 39.570942, 41.64385 ],
           dtype=float32)}



creates a dictionary for the upper and lower prediction interval with the following naming convention: model_{quantile_level}_predictions
- the key value changes respect to the quantile level specified during fit()


```python
pd.DataFrame(cqr_preds).head()
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
      <th>model_40_predictions</th>
      <th>model_60_predictions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23.486837</td>
      <td>38.722267</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18.673929</td>
      <td>33.717125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32.484924</td>
      <td>46.950207</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27.112062</td>
      <td>42.872467</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28.803766</td>
      <td>44.487522</td>
    </tr>
  </tbody>
</table>
</div>




```python
lower_quantile_preds = cqr_preds["model_40_predictions"]
upper_quantile_preds = cqr_preds["model_60_predictions"]
cqr_xgboost.plot_pi_line_graph(y_test, lower_quantile_preds, upper_quantile_preds)
print(f"symmetric coverage {cqr_xgboost.coverage(y_test, lower_quantile_preds, upper_quantile_preds)}")
print(f"symmetric average_width {cqr_xgboost.average_width(lower_quantile_preds, upper_quantile_preds)}")
print(f"symmetric cwc {cqr_xgboost.cwc(0.9, y_test, lower_quantile_preds, upper_quantile_preds)}")

```


    
![png](README_files/README_20_0.png)
    


    symmetric coverage 0.96
    symmetric average_width 15.935728073120117
    symmetric cwc 3.719721652657278e-12


### Also have the option for an asymmetric loss function which optimises the upper and lower quantiles separately


```python
# instantiate the model
asymmetric_cqr_xgboost = XGBoostCQR(model_params=xgboost_params,num_boost_round=int(100), alpha=0.90)

# grid search upper and lower quantiles
lower_qr_quantile, upper_qr_quantile, conformity_score = cqr_xgboost.cqr_grid_search_alpha(
    [0.1, 0.2, 0.4,],            # lower quantile coverage probability levels
    [0.6, 0.8, 0.9],             # upper quantile coverage probability levels
    X_train.values,         
    y_train.values, 
    X_calibration.values,   # unseen holdout calibration set used for the conformal prediction portion
    y_calibration.values, 
    X_test.values, 
    y_test.values, 
    0.25                    # split size for the training data of the initial quantile regression
    )
```

    Evaluating --- Lower alpha: 0.1 --- Upper alpha 0.6
    Evaluating --- Lower alpha: 0.1 --- Upper alpha 0.8
    Evaluating --- Lower alpha: 0.1 --- Upper alpha 0.9
    Evaluating --- Lower alpha: 0.2 --- Upper alpha 0.6
    Evaluating --- Lower alpha: 0.2 --- Upper alpha 0.8
    Evaluating --- Lower alpha: 0.2 --- Upper alpha 0.9
    Evaluating --- Lower alpha: 0.4 --- Upper alpha 0.6
    Evaluating --- Lower alpha: 0.4 --- Upper alpha 0.8
    Evaluating --- Lower alpha: 0.4 --- Upper alpha 0.9
    best CWC: 2.2979702687205084e-12
    best coverage: 0.944
    QR Alphas: [0.1, 0.6]
       l_alpha  u_alpha           cwc  coverage  average_PI_width
    0      0.1      0.6  2.297970e-12     0.968         20.983686
    1      0.1      0.8  5.785209e-12     0.944         16.820417
    2      0.1      0.9  7.639414e-12     0.936         15.943439
    3      0.2      0.6  5.581861e-12     0.944         17.615952
    4      0.2      0.8  3.708949e-12     0.960         16.003836
    5      0.2      0.9  6.238370e-12     0.944         15.047573
    6      0.4      0.6  4.770299e-12     0.952         15.728806
    7      0.4      0.8  3.616456e-12     0.960         16.588610
    8      0.4      0.9  6.010613e-12     0.944         15.938600


The Output for Grid Search shows key metrics (Coverage-Width Criterion, Coverage, average PI width) - best alphas recommended based on lowest CWC which changes based on arbitrary choice of eta parameter in CWC formula (higher eta favours high coverage => wider PIs, lower eta favours narrow PIs => lower coverage) thus for best results experiment with different values of eta or use the full output to also compare coverage and PI width when settling on lower and upper quantile levels. The balance of coverage and PI width will vary based on application


```python
# fit the model
asymmetric_cqr_xgboost.fit(
    X_train.values,
    y_train.values,
    X_calibration.values, 
    y_calibration.values, 0.25, 
    lower_qr_quantile=0.4, 
    upper_qr_quantile=0.6, 
    conformity_score_method="asymmetric"
    )
# make predictions
asym_cqr_preds = asymmetric_cqr_xgboost.predict(X_test)     
# access the lower and upper prediction interval values for the X_test data
asym_lower_quantile_preds = asym_cqr_preds["model_40_predictions"]
asym_upper_quantile_preds = asym_cqr_preds["model_60_predictions"]
# plot the prediction interval against the actual y_test values
asymmetric_cqr_xgboost.plot_pi_line_graph(y_test, asym_lower_quantile_preds, asym_upper_quantile_preds)
# print out some key metrics evaluating the prediction interval
print(f"asymmetric coverage {asymmetric_cqr_xgboost.coverage(y_test, asym_lower_quantile_preds, asym_upper_quantile_preds)}")
print(f"asymmetric average_width {asymmetric_cqr_xgboost.average_width(asym_lower_quantile_preds, asym_upper_quantile_preds)}")
print(f"asymmetric cwc {asymmetric_cqr_xgboost.cwc(0.9, y_test, asym_lower_quantile_preds, asym_upper_quantile_preds)}")
```

    Evaluating --- Lower QR quantile: 0.4 --- Upper QR quantile 0.6 --- CQR alpha 0.9
    {'model_40': <xgboost.core.Booster object at 0x000001A4DA37B290>, 'model_60': <xgboost.core.Booster object at 0x000001A4DA3794F0>}
    (np.float64(6.199072327530171), np.float64(7.433534967125333))



    
![png](README_files/README_24_1.png)
    


    asymmetric coverage 0.96
    asymmetric average_width 15.902301788330078
    asymmetric cwc 3.725008720146514e-12


Results in a marginally lower coverage probability but a narrower average PI width

### XGBoost Quantile Regression Demo


```python
qr_xgboost = XGBoostQuantileRegressor(
    model_params=xgboost_params,
    num_boost_round=int(100),
    early_stopping_rounds=10,
    quantiles=[0.05, 0.95]  # coverage probability for the conformal prediction part of CQR - should be set to your final desired coverage level
    )
qr_xgboost.fit(X_train, y_train, validation_size=0.25)
qr_preds = qr_xgboost.predict(X_test)
# access the lower and upper prediction interval values for the X_test data
lower_qr_preds = qr_preds["model_5_predictions"]
upper_qr_preds = qr_preds["model_95_predictions"]
# plot the prediction interval against the actual y_test values
asymmetric_cqr_xgboost.plot_pi_line_graph(y_test, lower_qr_preds, upper_qr_preds)
# print out some key metrics evaluating the prediction interval
print(f"asymmetric coverage {asymmetric_cqr_xgboost.coverage(y_test, lower_qr_preds, upper_qr_preds)}")
print(f"asymmetric average_width {asymmetric_cqr_xgboost.average_width(lower_qr_preds, upper_qr_preds)}")
print(f"asymmetric cwc {asymmetric_cqr_xgboost.cwc(0.9, y_test, lower_qr_preds, upper_qr_preds)}")
```


    
![png](README_files/README_27_0.png)
    


    asymmetric coverage 0.888
    asymmetric average_width 14.869775772094727
    asymmetric cwc 3.3716176135545774e-11



```python
bootstrap = XGBoostBootstrap(
    model_params={},
    num_boost_round=int(100),
    alpha=0.98
    )
bootstrap.fit(X_train, y_train, n_bootstrap=100, sample_size_ratio=0.7)
bs_preds = bootstrap.predict(X_test)
# access the lower and upper prediction interval values for the X_test data
lower_qr_preds = bs_preds["model_1_predictions"]
upper_qr_preds = bs_preds["model_99_predictions"]


# plot the prediction interval against the actual y_test values
bootstrap.plot_pi_line_graph(y_test, lower_qr_preds, upper_qr_preds)
# print out some key metrics evaluating the prediction interval
print(f"Bootstrap coverage {bootstrap.coverage(y_test, lower_qr_preds, upper_qr_preds)}")
print(f"Bootstrap average_width {bootstrap.average_width(lower_qr_preds, upper_qr_preds)}")
print(f"Bootstrap cwc {bootstrap.cwc(0.9, y_test, lower_qr_preds, upper_qr_preds)}")
```

    ----- Training model 1 / 100 -----
    ----- Training model 2 / 100 -----
    ----- Training model 3 / 100 -----
    ----- Training model 4 / 100 -----
    ----- Training model 5 / 100 -----
    ----- Training model 6 / 100 -----
    ----- Training model 7 / 100 -----
    ----- Training model 8 / 100 -----
    ----- Training model 9 / 100 -----
    ----- Training model 10 / 100 -----
    ----- Training model 11 / 100 -----
    ----- Training model 12 / 100 -----
    ----- Training model 13 / 100 -----
    ----- Training model 14 / 100 -----
    ----- Training model 15 / 100 -----
    ----- Training model 16 / 100 -----
    ----- Training model 17 / 100 -----
    ----- Training model 18 / 100 -----
    ----- Training model 19 / 100 -----
    ----- Training model 20 / 100 -----
    ----- Training model 21 / 100 -----
    ----- Training model 22 / 100 -----
    ----- Training model 23 / 100 -----
    ----- Training model 24 / 100 -----
    ----- Training model 25 / 100 -----
    ----- Training model 26 / 100 -----
    ----- Training model 27 / 100 -----
    ----- Training model 28 / 100 -----
    ----- Training model 29 / 100 -----
    ----- Training model 30 / 100 -----
    ----- Training model 31 / 100 -----
    ----- Training model 32 / 100 -----
    ----- Training model 33 / 100 -----
    ----- Training model 34 / 100 -----
    ----- Training model 35 / 100 -----
    ----- Training model 36 / 100 -----
    ----- Training model 37 / 100 -----
    ----- Training model 38 / 100 -----
    ----- Training model 39 / 100 -----
    ----- Training model 40 / 100 -----
    ----- Training model 41 / 100 -----
    ----- Training model 42 / 100 -----
    ----- Training model 43 / 100 -----
    ----- Training model 44 / 100 -----
    ----- Training model 45 / 100 -----
    ----- Training model 46 / 100 -----
    ----- Training model 47 / 100 -----
    ----- Training model 48 / 100 -----
    ----- Training model 49 / 100 -----
    ----- Training model 50 / 100 -----
    ----- Training model 51 / 100 -----
    ----- Training model 52 / 100 -----
    ----- Training model 53 / 100 -----
    ----- Training model 54 / 100 -----
    ----- Training model 55 / 100 -----
    ----- Training model 56 / 100 -----
    ----- Training model 57 / 100 -----
    ----- Training model 58 / 100 -----
    ----- Training model 59 / 100 -----
    ----- Training model 60 / 100 -----
    ----- Training model 61 / 100 -----
    ----- Training model 62 / 100 -----
    ----- Training model 63 / 100 -----
    ----- Training model 64 / 100 -----
    ----- Training model 65 / 100 -----
    ----- Training model 66 / 100 -----
    ----- Training model 67 / 100 -----
    ----- Training model 68 / 100 -----
    ----- Training model 69 / 100 -----
    ----- Training model 70 / 100 -----
    ----- Training model 71 / 100 -----
    ----- Training model 72 / 100 -----
    ----- Training model 73 / 100 -----
    ----- Training model 74 / 100 -----
    ----- Training model 75 / 100 -----
    ----- Training model 76 / 100 -----
    ----- Training model 77 / 100 -----
    ----- Training model 78 / 100 -----
    ----- Training model 79 / 100 -----
    ----- Training model 80 / 100 -----
    ----- Training model 81 / 100 -----
    ----- Training model 82 / 100 -----
    ----- Training model 83 / 100 -----
    ----- Training model 84 / 100 -----
    ----- Training model 85 / 100 -----
    ----- Training model 86 / 100 -----
    ----- Training model 87 / 100 -----
    ----- Training model 88 / 100 -----
    ----- Training model 89 / 100 -----
    ----- Training model 90 / 100 -----
    ----- Training model 91 / 100 -----
    ----- Training model 92 / 100 -----
    ----- Training model 93 / 100 -----
    ----- Training model 94 / 100 -----
    ----- Training model 95 / 100 -----
    ----- Training model 96 / 100 -----
    ----- Training model 97 / 100 -----
    ----- Training model 98 / 100 -----
    ----- Training model 99 / 100 -----
    ----- Training model 100 / 100 -----



    
![png](README_files/README_28_1.png)
    


    Bootstrap coverage 0.896
    Bootstrap average_width 7.640153408050537
    Bootstrap cwc 3.4321849985369016e-11


Bootstrap undercovers with the default parameters - introducing randomness via XGBoosts hyperparameters (subsample, colsample_bytree, colsample_bylevel  , ...) may reduce this effect


```python
bootstrap.plot_pi_line_graph(y_test, lower_qr_preds, upper_qr_preds, X_test["Feature1"])
```


    
![png](README_files/README_30_0.png)
    


###


```python
bootstrap.plot_coverage_probability_binned(y_test, lower_qr_preds, upper_qr_preds, x=X_test["Feature1"])
```


```python
bootstrap.plot_pi_width(lower_qr_preds, upper_qr_preds, x=X_test["Feature1"])
```
