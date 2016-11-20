from __future__ import print_function
import __builtin__

import time

from sklearn import linear_model
from sklearn.base import RegressorMixin, MetaEstimatorMixin, BaseEstimator
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from theano.gradient import np

from nn_regression import MLPRegressor
from nw_kernel_regression import KernelRegression


class EnsembleRegressor(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    # Static member variables
    _ensemble_regressors_auto = (
        linear_model.LinearRegression(fit_intercept=True),
        Pipeline(
            [('poly', PolynomialFeatures(degree=2)),
             ('linear', linear_model.LinearRegression(fit_intercept=False))]
        ),
        KernelRegression(kernel='poly'),
        DecisionTreeRegressor(max_depth=4),
        DecisionTreeRegressor(max_depth=None),
        RandomForestRegressor(n_estimators=100),
    )

    _ensemble_possible_regressors = (
        linear_model.LinearRegression(fit_intercept=True),
        Pipeline(
            [('poly', PolynomialFeatures(degree=2)),
             ('linear', linear_model.LinearRegression(fit_intercept=False))]
        ),
        # # linear_model.Ridge(alpha=4, fit_intercept=True),
        KernelRegression(kernel='poly'),
        # linear_model.RidgeCV(alphas=[.01, .1, .3, .5, 1], fit_intercept=True),
        # # linear_model.Lasso(alpha=4, fit_intercept=True),
        # linear_model.LassoCV(n_alphas=100, fit_intercept=True, max_iter=5000),
        # linear_model.ElasticNet(alpha=1),
        # linear_model.ElasticNetCV(n_alphas=100, l1_ratio=.5),
        # linear_model.OrthogonalMatchingPursuit(),
        # linear_model.BayesianRidge(),
        # # linear_model.ARDRegression(),
        # linear_model.SGDRegressor(),
        # # linear_model.PassiveAggressiveRegressor(loss='squared_epsilon_insensitive'),
        # linear_model.RANSACRegressor(),
        # LinearSVR(max_iter=1e4, fit_intercept=True, loss='squared_epsilon_insensitive', C=0.5),
        # SVR(max_iter=1e4, kernel='poly', C=1, degree=4),
        # SVR(max_iter=1e4, kernel='rbf', C=1, gamma=0.1),
        # SVR(kernel='linear', C=1),
        # SVR(kernel='linear', C=0.5),
        # SVR(kernel='linear', C=0.1),
        # DecisionTreeRegressor(max_depth=5),
        DecisionTreeRegressor(max_depth=4),
        DecisionTreeRegressor(max_depth=None),
        RandomForestRegressor(n_estimators=100),
        # AdaBoostRegressor(learning_rate=0.9, loss='square'),
        # BaggingRegressor(),
        MLPRegressor()
    )

    _ensemble_nn = [MLPRegressor(nb_epoch=1000) for _ in range(5)]  # 5 Multi Layer Perceptrons in the ensemble

    _ensemble_nn_large = [MLPRegressor(nb_epoch=500) for _ in range(10)]  # 5 Multi Layer Perceptrons in the ensemble
    _ensemble_nn_xlarge = [MLPRegressor(nb_epoch=500) for _ in range(30)]  # 5 Multi Layer Perceptrons in the ensemble

    _ensemble_ridge_regression = [
        linear_model.Ridge(alpha=alpha, fit_intercept=True, normalize=True)
        for alpha in np.arange(.1,1,.2)]  # 5 Ridge Regressors

    _ensemble_auto_large = (
        linear_model.LinearRegression(fit_intercept=True),
        Pipeline(
            [('poly', PolynomialFeatures(degree=2)),
             ('linear', linear_model.LinearRegression(fit_intercept=False))]
        ),
        linear_model.Ridge(alpha=0.5, fit_intercept=True, normalize=True),
        KernelRegression(kernel='poly'),
        # linear_model.RidgeCV(alphas=[.01, .1, .3, .5, 1], fit_intercept=True),
        linear_model.Lasso(alpha=0.1, fit_intercept=True),
        # linear_model.LassoCV(n_alphas=100, fit_intercept=True, max_iter=5000),
        # linear_model.ElasticNet(alpha=1),
        # linear_model.ElasticNetCV(n_alphas=100, l1_ratio=.5),
        linear_model.OrthogonalMatchingPursuit(),
        # linear_model.BayesianRidge(),
        # # linear_model.ARDRegression(),
        # linear_model.SGDRegressor(),
        # linear_model.PassiveAggressiveRegressor(loss='squared_epsilon_insensitive'),
        # linear_model.RANSACRegressor(),
        LinearSVR(max_iter=1e3, fit_intercept=True, loss='squared_epsilon_insensitive', C=1),
        SVR(max_iter=1e3, kernel='poly', C=1, degree=3),
        SVR(max_iter=1e3, kernel='rbf', C=1),
        SVR(max_iter=1e3, kernel='sigmoid', C=1),
        # SVR(kernel='linear', C=1),
        # SVR(kernel='linear', C=0.5),
        # SVR(kernel='linear', C=0.1),
        # DecisionTreeRegressor(max_depth=5),
        DecisionTreeRegressor(max_depth=4),
        DecisionTreeRegressor(max_depth=None),
        RandomForestRegressor(n_estimators=100),
        AdaBoostRegressor(learning_rate=0.9, loss='square'),
        BaggingRegressor(),
        # MLPRegressor(num_hidden_units=5)
    )

    # self._ensemble_nn = [MLPRegressor(num_hidden_units=(i+6), nb_epoch=(i+6)*100) for i in range(5)]  # 5 Multi Layer Perceptrons in the ensemble

    def __init__(self, type='auto', verbose=False):
        '''
        :param type: Possible values: 'auto', 'mlp', 'mlp_large', 'ridge', 'auto_large' (defaults to 'auto').
                     Choice of set of regressors, 'auto' will use various standard regressors (usually linear
                     regression, NW-kernel, decision trees and random forests, but subject to change).
                     'mlp' will use 5 Multi-Layer Perceptrons, each with 10 hidden units, batch_size=32 and 1000 epochs.
                     'mlp_large' will use 10 MLPs, each with 10 hidden units, batch_size=32 and only 500 epochs.
                     'ridge' will train 5 ridge regressors with different alphas.
        :param verbose:
        '''
        self._verbose = verbose
        self.type = type.lower() # convert type to lowercase

        if type == 'mlp':
            self.regressors = EnsembleRegressor._ensemble_nn
        elif type == 'mlp_large':
            self.regressors = EnsembleRegressor._ensemble_nn_large
        elif type == 'mlp_xlarge':
            self.regressors = EnsembleRegressor._ensemble_nn_xlarge
        elif type == 'ridge':
            self.regressors = EnsembleRegressor._ensemble_ridge_regression
        elif type == 'auto_large':
            self.regressors = EnsembleRegressor._ensemble_auto_large
        else:
            self.regressors = EnsembleRegressor._ensemble_regressors_auto


        # set regressor labels
        self.regressor_labels = []
        self.regressor_count = len(self.regressors)
        for i, regr in enumerate(self.regressors):
            self.regressor_labels.append(str(regr))

    def _dprint(self, *args, **kwargs):
        """overload print() function to only print when verbose=True."""
        if self._verbose:
            return __builtin__.print(*args, **kwargs)

    def fit(self, X_train, y_train, samples_per_regressor=None, regressor_overlap=0):
        """ Fits the model for all the regression algorithms in the ensemble.
            The models themselves can be accessed directly at EnsembleRegressor.regressors,
            and their labels is accessible in EnsembleRegressor.regressor_labels.

        :param X_train: Data matrix. Shape [# samples, # features].
        :param y_train: Target value vector.
        :param samples_per_regressor: Number of samples from X_train that each regressor will be trained on.
                                      Default 'None' will cause all regressors to be trained on all samples.
        :param regressor_overlap: If samples_per_regressor is not None, this is the number of samples overlapping for
                                  every adjacent pair of regressors. Defaults to no overlap.
        """
        start_sample = 0
        if samples_per_regressor is None:
            end_sample = None
        else:
            end_sample = samples_per_regressor

        start = time.time()
        for i, regr in enumerate(self.regressors):
            self._dprint('## ' + str(i) + '. ' + str(regr))

            X = X_train[start_sample:end_sample, :]
            y = y_train[start_sample:end_sample]
            regr.fit(X, y)

            if samples_per_regressor is not None:
                start_sample = start_sample + samples_per_regressor - regressor_overlap
                end_sample = start_sample + samples_per_regressor

            if type(regr) in [linear_model.LinearRegression, linear_model.Ridge, LinearSVR]:
                self._dprint('\tCoefficients: ', ', '.join(['%.2f' % f for f in regr.coef_]))

            if hasattr(regr, 'alphas_'):
                self._dprint('\tAlphas: ', ', '.join(['%.2f' % f for f in regr.alphas_]))

        self._dprint('Total running time: %.2f' % (time.time() - start))

    def predict(self, X):
        """
        :param X: Data matrix. Shape [# samples, # features].
        :return: Ensemble predictions. Shape [# regressors, # samples].
        """
        Z = np.ndarray(shape=(len(self.regressors), X.shape[0]))
        for i, regr in enumerate(self.regressors):
            # zip the real and predicted values together, sort them, and unzip them
            try:
                Z[i, :] = regr.predict(X)
            except:
                print(regr)
                raise

        return Z

    def score(self, X_test, y_test, **kwargs):
        """
        :return: vector with the R^2 score for each regressor
        """
        s = np.zeros(self.regressor_count)
        for i, regr in enumerate(self.regressors):
            try:
                s[i] = regr.score(X_test, y_test)
            except:
                print(regr)
                raise
        return s

    def mean_squared_error(self, X_test, y_test):
        """
        :return: vector with the MSE for each regressor
        """
        Z = self.predict(X_test)
        return np.mean((Z - y_test[None, :])**2, 1)
        # y[None, :] ensures that the vector is properly oriented
        # np.mean(..., 1) does the mean along the columns returning regressor_count results
