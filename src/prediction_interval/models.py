
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class PredictionIntervalResults:
    """
    class to store the results of a prediction interval model.
    provides simple graphing options to visualise the PI
    """
    
    def coverage(self, y_actual, y_lower, y_upper, epsilon=1e-5):
        """
        Calculates the coverage of the prediction intervals.

        Parameters
        ----------
        y_actual : array-like
            True values.
        y_lower : array-like
            Lower bounds of the prediction intervals.
        y_upper : array-like
            Upper bounds of the prediction intervals.
        epsilon : float, optional
            A small value to prevent division by zero (default is 1e-5).

        Returns
        -------
        float
            The coverage of the prediction intervals.
        """
        coverage = (y_actual + epsilon >= y_lower) & (y_actual - epsilon <= y_upper)
        return np.sum(coverage) / len(y_actual)
    

    def average_width(self, y_lower, y_upper):
        """
        Calculates the average width of the prediction intervals.

        Parameters
        ----------
        y_lower : array-like
            Lower bounds of the prediction intervals.
        y_upper : array-like
            Upper bounds of the prediction intervals.

        Returns
        -------
        float
            The average width of the prediction intervals.
        """
        return np.mean(abs(y_upper - y_lower))

    def normalised_average_width(self, y_actual, y_lower, y_upper):
        """
        Calculates the normalised average width of the prediction intervals.

        Parameters
        ----------
        y_actual : array-like
            True values.
        y_lower : array-like
            Lower bounds of the prediction intervals.
        y_upper : array-like
            Upper bounds of the prediction intervals        
        
        Returns
        -------
        float
            The average width of the prediction intervals.
        """
        y_max = y_actual.max()
        y_min = y_actual.min()
        R = y_max - y_min
        return np.mean(abs(y_upper - y_lower)) / R
    
    def cwc(self, alpha, y_actual, y_lower, y_upper, eta=30):
        """
        Coverage Width Criterion as formulated: "Ensemble Conformalized Quantile Regression for Probabilistic Time Series Forecasting"

        Parameters
        ----------
        alpha : float
            The desired coverage level.
        y_actual : array-like
            True values.
        y_lower : array-like
            Lower bounds of the prediction intervals.
        y_upper : array-like
            Upper bounds of the prediction intervals.
        eta : float, optional
            Penalty term for under-coverage (default is 30).

        Returns
        -------
        float
            The calculated CWC.
        """
        R = np.max(y_actual) - np.min(y_actual)
        n = y_actual.shape[0]
        pinaw = (1/(n*R)) * np.sum(y_upper - y_lower)       # Prediction Interval Normalized Average Width
        cwc = (1-pinaw) * np.exp(-eta * (self.coverage(y_actual, y_lower, y_upper) - (1-alpha)))
        return cwc

    def plot_pi_line_graph(self, y_test, lower_quantile_preds, upper_quantile_preds, x=None, plot_type="line"):
        """
        Plots a line graph of prediction intervals along with actual values.

        Parameters
        ----------
        y_test : array-like
            The actual observed values.
        lower_quantile_preds : array-like
            The predicted lower quantile values, representing the lower bounds of the prediction intervals.
        upper_quantile_preds : array-like
            The predicted upper quantile values, representing the upper bounds of the prediction intervals.
        x : array-like, optional
            The x-axis values. If None, a default range will be created.
        plot_type: str
            either "line" or "ribbon" to determine the type of plot created
        Returns
        -------
        None
            This method displays a plot and does not return any value.
        """
        if x is not None:
            assert len(x) == len(y_test) == len(lower_quantile_preds) == len(upper_quantile_preds), (
                f"Length mismatch: "
                f"x has length {len(x)}, "
                f"y_test has length {len(y_test)}, "
                f"lower_quantile_preds has length {len(lower_quantile_preds)}, "
                f"upper_quantile_preds has length {len(upper_quantile_preds)}. "
                "All inputs must have the same length."
            )
            if plot_type == "ribbon":
                # X values must be sorted for matplotlib fill_between
                # Ensure x is a Pandas Series
                if isinstance(x, pd.Series):
                    sorted_indices = np.argsort(x.values)  # Get sorting indices based on values
                    x = x.iloc[sorted_indices]  # Use .iloc for Pandas Series
                else:
                    sorted_indices = np.argsort(x)  # Standard sorting for NumPy arrays
                    x = x[sorted_indices]

                # Apply sorting to NumPy arrays
                y_test = y_test.iloc[sorted_indices] if isinstance(y_test, pd.Series) else y_test[sorted_indices]
                lower_quantile_preds = lower_quantile_preds.iloc[sorted_indices] if isinstance(lower_quantile_preds, pd.Series) else lower_quantile_preds[sorted_indices]
                upper_quantile_preds = upper_quantile_preds.iloc[sorted_indices] if isinstance(upper_quantile_preds, pd.Series) else upper_quantile_preds[sorted_indices]
        else:
            x = np.linspace(0, y_test.shape[0], y_test.shape[0])


        if plot_type not in ("line", "ribbon"):
            raise ValueError("Method must be 'line' or 'ribbon'.")

        # Setting the Seaborn style
        sns.set_style("white")
        
        # Create a figure and axis
        plt.figure(figsize=(15, 6))

        # Define x-axis values
        if plot_type == "line":
            sns.lineplot(x=x, y=lower_quantile_preds, label="Lower Quantile", color="blue", linestyle="--", linewidth=1)
            sns.lineplot(x=x, y=y_test.values, label="Actual Values", color="green", linestyle="-", linewidth=1)
            sns.lineplot(x=x, y=upper_quantile_preds, label="Upper Quantile", color="red", linestyle="--", linewidth=1)
        else:
            plt.fill_between(x, lower_quantile_preds, upper_quantile_preds, color="red", alpha=0.2, label="Prediction Interval")
            sns.lineplot(x=x, y=y_test.values, label="Actual Values", color="green", linestyle="-", linewidth=1)

        # Plot each line with improved colors and styles using Seaborn

        # Add labels and title
        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.title("Prediction Interval with Actual Values")

        # Add a legend with a better location
        plt.legend(loc="upper right")
        sns.despine()
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def plot_coverage_probability_binned(self, y_test, lower_quantile_preds, upper_quantile_preds, x=None, bins=50):
        """
        Plots the coverage proportion across binned sections of the x-axis.

        Parameters
        ----------
        y_test : array-like
            The actual observed values.
        lower_quantile_preds : array-like
            The predicted lower quantile values, representing the lower bounds of the prediction intervals.
        upper_quantile_preds : array-like
            The predicted upper quantile values, representing the upper bounds of the prediction intervals.
        x : array-like, optional
            The x-axis values. If None, a default range will be created.
        bins : int, optional
            The number of bins to group the x-axis values into (default is 10).

        Returns
        -------
        None
            Displays a plot showing the coverage proportion for each bin.
        """
        if x is None:
            x = np.arange(len(y_test))
        
        x, y_test = np.array(x), np.array(y_test)
        lower_quantile_preds, upper_quantile_preds = np.array(lower_quantile_preds), np.array(upper_quantile_preds)

        # Calculate coverage
        k_u = np.maximum(np.zeros_like(y_test), np.sign(upper_quantile_preds - y_test))
        k_l = np.maximum(np.zeros_like(y_test), np.sign(y_test - lower_quantile_preds))
        coverage = k_u * k_l

        # Bin x values
        bin_edges = np.linspace(np.min(x), np.max(x), bins + 1)
        bin_indices = np.digitize(x, bin_edges) - 1

        # Calculate coverage proportion per bin
        coverage_proportions = [np.mean(coverage[bin_indices == i]) if np.any(bin_indices == i) else np.nan for i in range(bins)]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot
        sns.set_style("white")
        plt.figure(figsize=(15, 6))
        plt.bar(bin_centers, coverage_proportions, width=(np.max(x) - np.min(x)) / bins, color="#1f1f8f", alpha=0.6, label="Coverage Proportion")
        
        plt.xlabel("X Value")
        plt.ylabel("Coverage Proportion")
        plt.title("Coverage Proportion Grouped by Binned X Values")
        plt.legend(loc="upper right")
        sns.despine()
        plt.grid(False)
        plt.tight_layout()
        plt.show()


    def plot_pi_width(self, lower_quantile_preds, upper_quantile_preds, x=None):
        """
        Plots the prediction interval width (upper - lower quantiles) over time.

        Parameters
        ----------
        lower_quantile_preds : array-like
            The predicted lower quantile values.
        upper_quantile_preds : array-like
            The predicted upper quantile values.
        x : array-like, optional
            The x-axis values. If None, a default range will be created.

        Returns
        -------
        None
            Displays a plot of PI width over time.
        """
        # Calculate the prediction interval width
        pi_width = upper_quantile_preds - lower_quantile_preds

        # Check if x is provided, else create a default range
        if x is None:
            x = np.linspace(0, len(lower_quantile_preds), len(lower_quantile_preds))

        # Plot the PI width over time
        sns.set_style("white")
        plt.figure(figsize=(15, 6))
        sns.lineplot(x=x, y=pi_width, label="PI Width", color="#1f1f8f", linestyle="-", linewidth=2)

        # Add labels and title
        plt.xlabel("Index")
        plt.ylabel("PI Width")
        plt.title("Prediction Interval Width Over Time")

        # Add a legend
        plt.legend(loc="upper right")
        sns.despine()
        plt.grid(False)
        plt.tight_layout()
        plt.show()

class XGBoostQuantileRegressor(PredictionIntervalResults):
    """
    A class used to perform quantile regression using XGBoost.
    - CQR: https://github.com/yromano/cqr/blob/master/cqr/torch_models.py
    Rounding in model means quantiles can be 2 decimal places maximum

    Attributes
    ----------
    model_params : dict
        Parameters for the XGBoost model.
    num_boost_round : int, optional
        Number of boosting rounds (default is 100).
    quantiles : np.array, optional
        Array of quantiles for which the models are to be trained (default is [0.05, 0.95]).
    early_stopping_rounds : int, optional
        Number of early stopping rounds (default is None).

    Methods
    -------
    predict(X_test, )
        Predicts the quantiles for the test data.
    fit(X_train, y_train, validation_size=0.25)
        Fits the quantile regression models to the training data.
    fit_quantile(x_data, y_data, quantile=0.5, validation_size=0.25)
        Fits a single quantile regression model.
    coverage(y_actual, y_lower, y_upper, epsilon=1e-5)
        Calculates the coverage of the prediction intervals.
    average_width(y_lower, y_upper)
        Calculates the average width of the prediction intervals.
    cwc(alpha, y_actual, y_lower, y_upper, eta=30)
        Calculates the Coverage Width Criterion (CWC).
    """

    def __init__(self, model_params: dict, num_boost_round=100, quantiles=np.array([0.05, 0.95]), early_stopping_rounds=None):
        """
        Initializes the XGBoostQuantileRegressor with the given parameters.
        Parameters:
        -----------
        model_params : dict
            Parameters for the XGBoost model.
        num_boost_round : int, optional
            Number of boosting rounds (default is 100).
        quantiles : np.array, optional
            Array of quantiles to be predicted (default is [0.05, 0.95]). Rounding in model means quantiles can be 2 decimal places maximum.
        early_stopping_rounds : int or None, optional
            Number of rounds for early stopping (default is None).
        """
        self.model_params = model_params
        self.num_boost_round = num_boost_round
        self.quantiles = quantiles
        self.early_stopping_rounds = early_stopping_rounds
        self.evals_result = {}
        self.models = {}


    def predict(self, X_test):
        """
        Predicts the quantiles for the test data using the trained models.

        Parameters
        ----------
        X_test : array-like
            Test data for which predictions are to be made.

        Returns
        -------
        dict
            Dictionary with the model predictions for each quantile.
        """
        if self.models == {}:
            raise RuntimeError("No trained models found. Run `fit()` first.")
        # have to use inplace_predict with quantile xgb
        model_predictions = {name+"_predictions": model.inplace_predict(X_test) for name, model in self.models.items()}
        return model_predictions
    

    def fit(self, X_train, y_train, validation_size=0.25) -> dict:
        """
        Fits the quantile regression models to the training data.

        Parameters
        ----------
        X_train : array-like
            Training data features.
        y_train : array-like
            Training data targets.
        validation_size : float, optional
            Proportion of the data to use for validation (default is 0.25).

        Returns
        -------
        dict
            Dictionary containing the quantile models fit on the training data.
        """
        self.models = {f"model_{str(int(quantile*100))}": self.fit_quantile(X_train, y_train, quantile, validation_size) for quantile in self.quantiles}
        return self.models


    def fit_quantile(self, x_data, y_data, quantile=0.5, validation_size=0.25):
        """
        Fits a single quantile regression model.

        Parameters
        ----------
        x_data : array-like
            Features for training the model.
        y_data : array-like
            Targets for training the model.
        quantile : float, optional
            The quantile to fit (default is 0.5).
        validation_size : float, optional
            Proportion of the data to use for validation (default is 0.25).

        Returns
        -------
        xgb.Booster
            Trained XGBoost model.
        """
        if not (0 < quantile < 1):
            raise ValueError("Quantile must be between 0 and 1.")
        X_train, X_test, y_train, y_validation = train_test_split(x_data, y_data, test_size=validation_size)
        # fit the quantile regression xgboost model
        self.model_params["objective"] =  "reg:quantileerror"
        self.model_params["tree_method"] =  "hist"
        self.model_params["quantile_alpha"] =  quantile

        QD_train = xgb.QuantileDMatrix(X_train, label=y_train)
        QD_validation = xgb.QuantileDMatrix(X_test, label=y_validation, ref=QD_train)

        model = xgb.train(
            params=self.model_params,
            dtrain=QD_train,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds = self.early_stopping_rounds,
            evals=[(QD_train, "train"), (QD_validation, "eval")],
            evals_result=self.evals_result,
            verbose_eval=0
            )
        return model

    def save(self, filepath):
        """Saves the model, conformity scores, and relevant metadata."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                "models": {name: model.save_raw() for name, model in self.models.items()},
                "lower_qr_quantile": self.lower_qr_quantile,
                "upper_qr_quantile": self.upper_qr_quantile,
                "alpha": self.alpha
            }, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Loads the model, conformity scores, and relevant metadata."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.lower_qr_quantile = data["lower_qr_quantile"]
            self.upper_qr_quantile = data["upper_qr_quantile"]
            self.alpha = data["alpha"]
            self.models = {}
            for name, raw_model in data["models"].items():
                booster = xgb.Booster()
                booster.load_model(raw_model)
                self.models[name] = booster
        print(f"Model loaded from {filepath}")



class XGBoostCQR(XGBoostQuantileRegressor):
    """
    A class to perform Conformalized Quantile Regression (CQR) using XGBoost.

    Attributes
    ----------
    alpha : float
        Conformal prediction alpha, not necessarily the same as quantile regression alpha.

    Methods
    -------
    symmetric_conformity_score(calib_actual, calib_lower, calib_upper)
        Calculates symmetric conformity scores.
    asymmetric_conformity_score(calib_actual, calib_lower, calib_upper)
        Calculates asymmetric conformity scores.
    cqr_grid_search_alpha(qr_lq_grid, qr_uq_grid, X_train, y_train, x_calibration, y_calibration, X_test, y_test, validation_size=0.25, conformity_score_method="symmetric")
        Performs grid search to find the best alpha values for the lower and upper quantiles.
    fit(X_train, y_train, x_calibration, y_calibration, validation_size=0.25, lower_qr_quantile=float, upper_qr_quantile=float, conformity_score_method="symmetric")
        Fits the CQR model.
    predict(X_test)
        Predicts using the CQR model.
    """

    def __init__(self, model_params: dict, num_boost_round=100, early_stopping_rounds=None, alpha=0.90):
        """
        Initializes the XGBoostCQR with the given parameters.
        Parameters:
        -----------
        model_params : dict
            Parameters for the XGBoost model.
        num_boost_round : int, optional
            Number of boosting rounds (default is 100).
        early_stopping_rounds : int or None, optional
            Number of rounds for early stopping (default is None).
        alpha : float, optional
            Conformal prediction alpha (default is 0.90).
        """
        super().__init__(model_params, num_boost_round, early_stopping_rounds)
        self.alpha = alpha      # conformal prediction alpha not necessarily the same as quantile regression alpha
        

    def symmetric_conformity_score(self, calib_actual, calib_lower, calib_upper):
        """
        Calculates symmetric conformity scores using the calibration set predictions - providing finite sample coverage guarantee
        - calibration set should be an unseen holdout set.

        Parameters
        ----------
        calib_actual : array-like
            Actual values from the calibration set.
        calib_lower : array-like
            Lower predictions from the calibration set.
        calib_upper : array-like
            Upper predictions from the calibration set.

        Returns
        -------
        tuple
            Conformity scores for the lower and upper bounds.
        """
        assert calib_actual.shape[0] == calib_lower.shape[0] == calib_upper.shape[0], "Input arrays must have the same length."
        E = np.maximum(calib_lower - calib_actual, calib_actual - calib_upper)
        m = calib_actual.shape[0]

        E_sorted = np.sort(E)    # sort in ascending order
        index = np.clip(int(np.ceil((self.alpha) * (m + 1))) - 1, 0, m - 1)

        conformity_score = E_sorted[max(0, min(index, m - 1))]

        return conformity_score, conformity_score

    def asymmetric_conformity_score(self, calib_actual, calib_lower, calib_upper):
        """
        Calculates asymmetric conformity scores using the calibration set - calibration set should be an unseen holdout set.

        Parameters
        ----------
        calib_actual : array-like
            Actual values from the calibration set.
        calib_lower : array-like
            Lower predictions from the calibration set.
        calib_upper : array-like
            Upper predictions from the calibration set.

        Returns
        -------
        tuple
            Asymmetric conformity scores for the lower and upper bounds.
        """
        assert calib_actual.shape[0] == calib_lower.shape[0] == calib_upper.shape[0], "Input arrays must have the same length."
        m = calib_actual.shape[0]
        # Calculate asymmetric errors for lower and upper quantiles
        E_low = calib_lower - calib_actual  # Asymmetric error for lower quantile
        E_high = calib_actual - calib_upper  # Asymmetric error for upper quantile

        E_low_sorted = np.sort(E_low)    # sort in ascending order
        E_high_sorted = np.sort(E_high)    # sort in ascending order

        index = np.clip(int(np.ceil(((1 + self.alpha) / 2) * (m + 1))) - 1, 0, m - 1)       # ensures index within bounds
        conformity_score_low = E_low_sorted[index]
        conformity_score_high = E_high_sorted[index]

        return conformity_score_low, conformity_score_high
    

    def cqr_grid_search_alpha(self, qr_lq_grid, qr_uq_grid, X_train, y_train, x_calibration, y_calibration,
                              X_test, y_test, validation_size=0.25, conformity_score_method="symmetric",
                              eta=30):
        """
        Performs grid search to find the best alpha values for the lower and upper quantiles.

        Parameters
        ----------
        qr_lq_grid : list
            List of lower quantile values to be tested.
        qr_uq_grid : list
            List of upper quantile values to be tested.
        X_train : array-like
            Training data features.
        y_train : array-like
            Training data targets.
        x_calibration : array-like
            Calibration data features.
        y_calibration : array-like
            Calibration data targets.
        X_test : array-like
            Test data features.
        y_test : array-like
            Test data targets.
        validation_size : float, optional
            Proportion of the data to use for validation (default is 0.25).
        conformity_score_method : str, optional
            Method for calculating conformity scores, either "symmetric" or "asymmetric" (default is "symmetric").
        eta: int, optional
            Used in the CWC metric which scores performance - balances penalty between coverage and average PI width
            higher eta favours high coverage => wider PIs
            lower eta favours narrow PIs => lower coverage

        Returns
        -------
        tuple
            best lower quantile, best upper quantile, and conformity score.
        """
        get_conformity_score = self.symmetric_conformity_score if conformity_score_method == "symmetric" else self.asymmetric_conformity_score
        best_cwc = np.inf
        best_conformity = None
        results_list = []

        for l_alpha in qr_lq_grid:
            for u_alpha in qr_uq_grid:
                print(f"Evaluating --- Lower alpha: {l_alpha} --- Upper alpha {u_alpha}")
                # creating QR
                qr = XGBoostQuantileRegressor(self.model_params, self.num_boost_round, [l_alpha, u_alpha], early_stopping_rounds=self.early_stopping_rounds)
                models = qr.fit(X_train, y_train, validation_size)

                # predicting calibration set using QR
                calibration_preds = qr.predict(x_calibration)
                lq_calib_preds_array, uq_calib_preds_array = calibration_preds[f"model_{str(int(l_alpha*100))}_predictions"], calibration_preds[f"model_{str(int(u_alpha*100))}_predictions"]
                
                # calculating conformity scores
                conformity_score_low, conformity_score_high = get_conformity_score(y_calibration, lq_calib_preds_array, uq_calib_preds_array)

                # calculating the CQR bounds on test data to evaluate the alpha combination
                test_preds = qr.predict(X_test)
                test_preds[f"model_{str(int(l_alpha*100))}_predictions"] -= conformity_score_low
                test_preds[f"model_{str(int(u_alpha*100))}_predictions"] += conformity_score_high

                # evaluating PI using CWC
                cwc =  self.cwc(self.alpha, y_test, test_preds[f"model_{str(int(l_alpha*100))}_predictions"], test_preds[f"model_{str(int(u_alpha*100))}_predictions"])
                coverage = self.coverage(y_test, test_preds[f"model_{str(int(l_alpha*100))}_predictions"], test_preds[f"model_{str(int(u_alpha*100))}_predictions"])
                av_width = self.average_width(test_preds[f"model_{str(int(l_alpha*100))}_predictions"], test_preds[f"model_{str(int(u_alpha*100))}_predictions"])
                # Append the results to the DataFrame
                results_list.append({"l_alpha": l_alpha, "u_alpha": u_alpha, "cwc": cwc, "coverage": coverage, "average_PI_width": av_width})

                # store info of models with the best CWC - need to check what is actually needed to save
                if cwc < best_cwc:
                    best_cwc = cwc
                    self.lower_qr_quantile = l_alpha
                    self.upper_qr_quantile = u_alpha
                    # best_models = deepcopy(models)
                    best_conformity = (conformity_score_low, conformity_score_high)
        # self.models = best_models
        self.conformity_score = best_conformity
        self.grid_search_alpha_results = pd.DataFrame(results_list)
        
        print(f"best CWC: {best_cwc}")
        print(f"best coverage: {coverage}")
        print(f"QR Alphas: [{self.lower_qr_quantile}, {self.upper_qr_quantile}]")
        print(self.grid_search_alpha_results)
        return self.lower_qr_quantile, self.upper_qr_quantile, self.conformity_score
        

    def fit(self, X_train, y_train, x_calibration, y_calibration, validation_size=0.25,
            lower_qr_quantile=float, upper_qr_quantile=float, conformity_score_method="symmetric"):
        """
        Fits the CQR model using the specified quantiles and conformity score method.
        Conformal regression alpha level defined in class creation, lower and upper quantile values in method refer to the quantile regression parameters

        Parameters
        ----------
        X_train : array-like
            Training data features.
        y_train : array-like
            Training data targets.
        x_calibration : array-like
            Calibration data features.
        y_calibration : array-like
            Calibration data targets.
        validation_size : float, optional
            Proportion of the data to use for validation (default is 0.25).
        lower_qr_quantile : float
            Lower quantile for the quantile regression part.
        upper_qr_quantile : float
            Upper quantile for the quantile regression part.
        conformity_score_method : str, optional
            Method for calculating conformity scores, either "symmetric" or "asymmetric" (default is "symmetric").
        """
        get_conformity_score = self.symmetric_conformity_score if conformity_score_method == "symmetric" else self.asymmetric_conformity_score

        self.lower_qr_quantile = lower_qr_quantile
        self.upper_qr_quantile = upper_qr_quantile
        print(f"Evaluating --- Lower QR quantile: {self.lower_qr_quantile} --- Upper QR quantile {self.upper_qr_quantile} --- CQR alpha {self.alpha}")

        # creating QR
        qr = XGBoostQuantileRegressor(self.model_params, self.num_boost_round, [self.lower_qr_quantile, self.upper_qr_quantile], early_stopping_rounds=self.early_stopping_rounds)
        self.models = qr.fit(X_train, y_train, validation_size)

        # predicting calibration set using QR
        calibration_preds = qr.predict(x_calibration)
        lq_calib_preds_array, uq_calib_preds_array = calibration_preds[f"model_{str(int(self.lower_qr_quantile*100))}_predictions"], calibration_preds[f"model_{str(int(self.upper_qr_quantile*100))}_predictions"]
                
        # calculating conformity scores
        lower_conformity_score, upper_conformity_score = get_conformity_score(y_calibration, lq_calib_preds_array, uq_calib_preds_array)

        self.conformity_score = (lower_conformity_score, upper_conformity_score)


    def predict(self, X_test):
        """
        Predicts using the CQR model.

        Parameters
        ----------
        X_test : array-like
            Test data for which predictions are to be made.

        Returns
        -------
        dict
            Dictionary with the model predictions for each quantile, adjusted for conformity.
        """
        print(self.models)
        if self.models == {}:
            raise RuntimeError("No trained models found. Run `fit()` first.")
        # returns a dictonary with the models predictions for each quantile 
        # have to use inplace_predict with quantile xgb
        model_predictions = {name+"_predictions": model.inplace_predict(X_test) for name, model in self.models.items()}
        model_predictions[f"model_{str(int(self.lower_qr_quantile*100))}_predictions"] -= self.conformity_score[0]
        model_predictions[f"model_{str(int(self.upper_qr_quantile*100))}_predictions"] += self.conformity_score[1]
        print(self.conformity_score)
        return model_predictions

    def save(self, filepath):
        """Saves the model, conformity scores, and relevant metadata."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                "models": {name: model.save_raw() for name, model in self.models.items()},
                "conformity_score": self.conformity_score,
                "lower_qr_quantile": self.lower_qr_quantile,
                "upper_qr_quantile": self.upper_qr_quantile,
                "alpha": self.alpha
            }, f)
        print(f"Model saved to {filepath}")
 
    def load(self, filepath):
        """Loads the model, conformity scores, and relevant metadata."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.conformity_score = data["conformity_score"]
            self.lower_qr_quantile = data["lower_qr_quantile"]
            self.upper_qr_quantile = data["upper_qr_quantile"]
            self.alpha = data["alpha"]
            self.models = {}
            for name, raw_model in data["models"].items():
                booster = xgb.Booster()
                booster.load_model(raw_model)
                self.models[name] = booster
        print(f"Model loaded from {filepath}")


class XGBoostBootstrap(PredictionIntervalResults):
    def __init__(self, model_params:dict, num_boost_round=100, method="bootstrap", alpha=90) -> None:
        """
        The constructor initialises the BootstrapStrategy object with the model, training, and testing data, as well as the target column name.
        For creating a Bootstrapped PI around XGBoost regressor model
        Parameters:
            model_params =  parameters for xgb.train() 
            num_boost_round =  Number of boosted rounds for xgboost models
            method: Specifies whether to use models trained with "bootstrap" or "monte_carlo" (default is "bootstrap").
            alpha =  The confidence level for the prediction interval (default is 90).
        -----------.
        """
        self.model_params = model_params
        self.num_boost_round = num_boost_round
        self.method = method
        self.alpha = alpha
        self.bootstrap_models_list = []  # Store bootstrap models
        self.monte_carlo_models_list = []  # Store MC ensemble models


    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, n_bootstrap: int = 100, sample_size_ratio: float = 1.0):
        """
        Trains multiple models using either the bootstrap resampling or Monte Carlo method.
        For Monte Carlo ensure subsample, colsample_bytree, colsample_bylevel are not 1 to ensure there is randomness between models in ensemble
        Parameters:
        ------------
            n_bootstrap: Number of models to train (default is 100).
            sample_size_ratio: FOR BOOSTRAPPING - The proportion of the dataset used for each resampled dataset (default is 1.0, meaning the full dataset is used).
            X_train : array-like, Training data features.
            y_train : array-like 1-D, Training data targets.
        """
        self.bootstrap_models_list = []
        self.monte_carlo_models_list = []


        n_samples = int(X_train.shape[0] * sample_size_ratio)

        for i in range(n_bootstrap):
            print(f"----- Training model {i+1} / {n_bootstrap} -----")

            if self.method == "bootstrap":
                indices = resample(range(len(X_train)), n_samples=n_samples, replace=True)
                X_resampled = X_train.iloc[indices]
                y_resampled = y_train.iloc[indices]

            elif self.method == "monte_carlo":
                X_resampled = X_train
                y_resampled = y_train
            else:
                raise ValueError("Method must be 'bootstrap' or 'monte_carlo'.")

            model = xgb.train(params=self.model_params, 
                              dtrain=xgb.DMatrix(X_resampled, label=y_resampled), 
                              num_boost_round=self.num_boost_round, 
                              verbose_eval=0)
            
            if self.method == "bootstrap":
                self.bootstrap_models_list.append(model)
            else:
                self.monte_carlo_models_list.append(model)

    def predict(self, X_test):
        """
        Predicts the target values using the trained models and computes the prediction interval (upper and lower bounds) based on the specified percentile.
        Parameters:
        ------------
            X_train : array-like, test data features.
        """
        if self.method == "bootstrap":
            models = self.bootstrap_models_list
        elif self.method == "monte_carlo":
            models = self.monte_carlo_models_list
        else:
            raise ValueError("Method must be 'bootstrap' or 'monte_carlo'.")

        if not models:
            raise RuntimeError("No trained models found. Run `fit()` first.")

        dtest = xgb.DMatrix(X_test)
        predictions = np.column_stack([model.predict(dtest) for model in models])

        # for percentile need to convert alpha from decimal to percentage
        lower_bound = np.percentile(predictions, (100 - (self.alpha*100)) / 2, axis=1)
        upper_bound = np.percentile(predictions, 50 + ((self.alpha*100) / 2), axis=1)
        median_predictions = np.percentile(predictions, 50, axis=1)

        return {
            f"model_{str(int(((100 - (self.alpha*100)) / 2)))}_predictions": lower_bound,
            f"mean_predictions": predictions.mean(axis=1),
            f"model_50_predictions": median_predictions,
            f"model_{str(int(50 + ((self.alpha*100) / 2)))}_predictions": upper_bound,
        }

    def save(self, filepath):
        """Saves the model, conformity scores, and relevant metadata."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                "models": {name: model.save_raw() for name, model in self.models.items()},
                "conformity_score": self.conformity_score,
                "lower_qr_quantile": self.lower_qr_quantile,
                "upper_qr_quantile": self.upper_qr_quantile,
                "alpha": self.alpha
            }, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Loads the model, conformity scores, and relevant metadata."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.conformity_score = data["conformity_score"]
            self.lower_qr_quantile = data["lower_qr_quantile"]
            self.upper_qr_quantile = data["upper_qr_quantile"]
            self.alpha = data["alpha"]
            self.models = {}
            for name, raw_model in data["models"].items():
                booster = xgb.Booster()
                booster.load_model(raw_model)
                self.models[name] = booster
        print(f"Model loaded from {filepath}")

