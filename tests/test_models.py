from src.prediction_interval.models import XGBoostQuantileRegressor, XGBoostCQR
import pytest
import numpy as np
import xgboost as xgb

# To use run pytest tests/test_models.py

@pytest.fixture
def sample_data():
    # Create a small synthetic dataset for testing
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    return X, y

@pytest.fixture
def model_params():
    # Default model parameters for XGBoost
    return {"eta": 0.1, "max_depth": 3}


def test_initialization(model_params):
    model = XGBoostQuantileRegressor(model_params)
    assert model.model_params == model_params
    assert model.num_boost_round == 100
    assert np.array_equal(model.quantiles, np.array([0.05, 0.95]))
    assert model.early_stopping_rounds is None
    assert model.models == {}



def test_fit(sample_data, model_params):
    X, y = sample_data
    model = XGBoostQuantileRegressor(model_params)
    models = model.fit(X, y)
    assert len(models) == 2  # Assuming two quantiles
    assert all(isinstance(m, xgb.Booster) for m in models.values())

def test_fit_invalid_quantile(sample_data, model_params):
    X, y = sample_data
    model = XGBoostQuantileRegressor(model_params, quantiles=np.array([1.5]))
    with pytest.raises(ValueError, match="Quantile must be between 0 and 1."):
        model.fit(X, y)



def test_predict(sample_data, model_params):
    X, y = sample_data
    model = XGBoostQuantileRegressor(model_params)
    model.fit(X, y)
    predictions = model.predict(X)
    assert isinstance(predictions, dict)
    assert len(predictions) == 2  # Assuming two quantiles
    for pred in predictions.values():
        assert pred.shape[0] == X.shape[0]

def test_predict_without_fit(sample_data, model_params):
    X, _ = sample_data
    model = XGBoostQuantileRegressor(model_params)
    with pytest.raises(RuntimeError):
        model.predict(X)



def test_coverage(sample_data, model_params):
    X, y = sample_data
    model = XGBoostQuantileRegressor(model_params)
    model.fit(X, y)
    predictions = model.predict(X)
    lower = predictions["model_5_predictions"]
    upper = predictions["model_95_predictions"]
    coverage = model.coverage(y, lower, upper)
    assert 0 <= coverage <= 1

def test_average_width(sample_data, model_params):
    X, y = sample_data
    model = XGBoostQuantileRegressor(model_params)
    model.fit(X, y)
    predictions = model.predict(X)
    lower = predictions["model_5_predictions"]
    upper = predictions["model_95_predictions"]
    avg_width = model.average_width(lower, upper)
    assert avg_width >= 0


def test_cqr_initialization(model_params):
    model = XGBoostCQR(model_params)
    assert model.alpha == 0.90
    assert model.model_params == model_params
    assert model.num_boost_round == 100

def test_symmetric_conformity_score():
    calib_actual = np.array([1, 2, 3])
    calib_lower = np.array([0.8, 1.8, 2.8])
    calib_upper = np.array([1.2, 2.2, 3.2])
    model = XGBoostCQR({})    # default alpha=0.9
    score_low, score_high = model.symmetric_conformity_score(calib_actual, calib_lower, calib_upper)
    # expect scores to be less than 0 as all actual points are with lower upper bound
    assert score_low <=0
    assert score_high <= 0

def test_asymmetric_conformity_score():
    calib_actual = np.array([1, 2, 3])
    calib_lower = np.array([0.8, 1.8, 2.8])
    calib_upper = np.array([1.2, 2.2, 3.2])
    model = XGBoostCQR({})    # default alpha=0.9
    score_low, score_high = model.asymmetric_conformity_score(calib_actual, calib_lower, calib_upper)
    assert score_low <= 0
    assert score_high <= 0

def test_cqr_grid_search_alpha(sample_data, model_params):
    X, y = sample_data
    model = XGBoostCQR(model_params)
    qr_lq_grid = [0.05, 0.10]
    qr_uq_grid = [0.90, 0.95]
    l_alpha, u_alpha, conformity_score = model.cqr_grid_search_alpha(
        qr_lq_grid, qr_uq_grid, X, y, X, y, X, y
    )
    assert 0 <= l_alpha <= 1
    assert 0 <= u_alpha <= 1
    assert isinstance(conformity_score, tuple)


def test_cqr_fit(sample_data, model_params):
    X, y = sample_data
    model = XGBoostCQR(model_params)
    model.fit(X, y, X, y, lower_qr_quantile=0.05, upper_qr_quantile=0.95)
    assert model.models is not None
    assert model.conformity_score is not None

def test_cqr_predict(sample_data, model_params):
    X, y = sample_data
    model = XGBoostCQR(model_params)
    model.fit(X, y, X, y, lower_qr_quantile=0.05, upper_qr_quantile=0.95)
    predictions = model.predict(X)
    assert isinstance(predictions, dict)
    assert "model_5_predictions" in predictions
    assert "model_95_predictions" in predictions
    assert predictions["model_5_predictions"].shape[0] == X.shape[0]
    assert predictions["model_95_predictions"].shape[0] == X.shape[0]


def test_asymmetric_conformity_score_mixed():
    calib_actual = np.array([100, 200, 300, 400, 500])
    calib_lower = np.array([90, 220, 290, 410, 480])  # Mix of lower and higher bounds
    calib_upper = np.array([110, 190, 310, 385, 520])
    
    # Deviation: lower-actual, actual-upper => largest lower deviation = 20 (220-200), largest upper deviation = 15 (400-385)
    expected_conformity_score_low = 20
    expected_conformity_score_high = 15
    
    model = XGBoostCQR({}, alpha=1)    # default alpha=0.9
    score_low, score_high = model.asymmetric_conformity_score(calib_actual, calib_lower, calib_upper)
    
    assert score_low == expected_conformity_score_low
    assert score_high == expected_conformity_score_high

def test_sym_conformity_score_large_sample_dataset():
    calib_actual = [ 4.97424837e-01, -8.87924111e-01, -6.58872911e-01,  4.22165833e-01,
                    -2.00114148e-01, -6.64231995e-01,  2.61461590e-01,  7.25062014e-01,
                    1.28163296e-01,  1.33275349e+00, -1.54935310e+00, -4.48061010e-01,
                    -8.98905052e-01, -5.84695829e-01, -1.81172935e+00,  7.55863490e-02,
                    -6.26276753e-01,  1.36017539e-01,  8.13160871e-02, -7.56164957e-01,
                    7.12145718e-01,  1.31609338e-01, -6.16746732e-01, -4.38562734e-01,]
    calib_lower = [-0.3412692,  -0.9344709,  -1.5359459,  -1.0109919,  -1.5473638,  -1.5566149,
                -0.12009155, -1.1922419,  -0.2735057,  -0.3824054,  -1.4905941,  -0.68552524,
                -0.8952529,  -1.2822505,  -1.4710933,  -0.8237698,  -1.0771097,  -1.2230061,
                -0.2802157,  -1.1301056,  -0.5753897,  -0.3851345,  -0.96862936, -1.4204015,]
    calib_upper = [ 5.77365696e-01, -1.12102414e-03, -5.46385825e-01, -1.72343686e-01,
                    1.20061345e-01, -1.17578693e-01,  5.28151035e-01,  3.59687597e-01,
                    1.77164823e-01,  8.12812269e-01, -4.81830209e-01,  6.73749372e-02,
                    5.29558510e-02, -3.41476411e-01, -5.18193781e-01,  1.97080582e-01,
                    -2.73129910e-01,  2.53314525e-01,  6.11782253e-01,  1.41620025e-01,
                    2.84556538e-01,  8.00889552e-01, -2.93089986e-01, -3.06418359e-01,]
    calib_upper = np.array(calib_upper) 
    calib_lower = np.array(calib_lower)
    calib_actual = np.array(calib_actual)
    model = XGBoostCQR({})    # default alpha=0.9
    score_low, score_high = model.symmetric_conformity_score(calib_actual, calib_lower, calib_upper)
    
    # Check that scores are non-negative
    assert np.isclose(score_low, 0.51994122)
    assert np.isclose(score_high, 0.51994122)
def test_asym_conformity_score_large_sample_dataset():
    calib_actual = [ 4.97424837e-01, -8.87924111e-01, -6.58872911e-01,  4.22165833e-01,
                    -2.00114148e-01, -6.64231995e-01,  2.61461590e-01,  7.25062014e-01,
                    1.28163296e-01,  1.33275349e+00, -1.54935310e+00, -4.48061010e-01,
                    -8.98905052e-01, -5.84695829e-01, -1.81172935e+00,  7.55863490e-02,
                    -6.26276753e-01,  1.36017539e-01,  8.13160871e-02, -7.56164957e-01,
                    7.12145718e-01,  1.31609338e-01, -6.16746732e-01, -4.38562734e-01,]
    calib_lower = [-0.3412692,  -0.9344709,  -1.5359459,  -1.0109919,  -1.5473638,  -1.5566149,
                -0.12009155, -1.1922419,  -0.2735057,  -0.3824054,  -1.4905941,  -0.68552524,
                -0.8952529,  -1.2822505,  -1.4710933,  -0.8237698,  -1.0771097,  -1.2230061,
                -0.2802157,  -1.1301056,  -0.5753897,  -0.3851345,  -0.96862936, -1.4204015,]
    calib_upper = [ 5.77365696e-01, -1.12102414e-03, -5.46385825e-01, -1.72343686e-01,
                    1.20061345e-01, -1.17578693e-01,  5.28151035e-01,  3.59687597e-01,
                    1.77164823e-01,  8.12812269e-01, -4.81830209e-01,  6.73749372e-02,
                    5.29558510e-02, -3.41476411e-01, -5.18193781e-01,  1.97080582e-01,
                    -2.73129910e-01,  2.53314525e-01,  6.11782253e-01,  1.41620025e-01,
                    2.84556538e-01,  8.00889552e-01, -2.93089986e-01, -3.06418359e-01,]
    
    calib_upper = np.array(calib_upper)
    calib_lower = np.array(calib_lower)
    calib_actual = np.array(calib_actual)
    model = XGBoostCQR({})    # default alpha=0.9
    score_low, score_high = model.asymmetric_conformity_score(calib_actual, calib_lower, calib_upper)
    
    # Check that scores are non-negative
    assert np.isclose(score_low, 0.34063605)
    assert np.isclose(score_high, 0.59450952)