import errno
import numpy as np
import scipy.io as sio
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.base import _partition_estimators
from sklearn.externals.joblib import Parallel, delayed
import types

from regression_datasets import DatasetFactory


def _parallel_helper(obj, methodname, *args, **kwargs):
    """Private helper to workaround Python 2 pickle limitations"""
    return getattr(obj, methodname)(*args, **kwargs)


def forest_regressor_predict(self, X):
    """Copy of the RandomForest Regression predict code, while retaining the entire ensemble in self.all_y_hat """
    # Check data
    X = self._validate_X_predict(X)

    # Assign chunk of trees to jobs
    n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

    # Parallel loop
    all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                         backend="threading")(
        delayed(_parallel_helper)(e, 'predict', X, check_input=False)
        for e in self.estimators_)

    # Reduce
    y_hat = sum(all_y_hat) / len(self.estimators_)

    """ This is the ONLY line changed: Save the decision tree results to the object for external use """
    self.all_y_hat = all_y_hat

    return y_hat


def main():
    # Create the RandomForest regressor, and replace its predict function with
    # a predict that saves the individual regressor outputs to a member variable all_y_hat
    regr = RandomForestRegressor(n_estimators=50)
    regr.predict = types.MethodType(forest_regressor_predict, regr)

    # Load Dataset
    #ds = DatasetFactory.friedman1(n_samples=20200)
    #ds = DatasetFactory.friedman2(n_samples=20200)
    ds = DatasetFactory.friedman3(n_samples=20200)
    Xtrain, X, ytrain, y = cross_validation.train_test_split(
                                    ds.data,ds.target,train_size=200,random_state=0)

    # Fit the model
    regr.fit(Xtrain, ytrain)
    yhat = regr.predict(X)
    Z = np.array(regr.all_y_hat)

    # Save results
    sio.savemat('RandomForestTest_Friedman3.mat', {
        'names': str(regr.estimators_),
        'Z': Z, 'y': y,
        'y_RandomForest': yhat,
        # 'Ztrain': Z_train, # NOTE: Combing DecisionTrees is unsupervised by nature
        #'ytrain': ytrain,
        'samples_per_regressor': 200,
        'regressor_samples_overlap': 200,
        'Ey': np.mean(y),
        'Ey2': np.mean(y ** 2),
        'Description': 'Random Forest Ensemble Test (Friedman #3)'
    })

if __name__ == "__main__":
    try:
        main()
        print('Done.')
    except KeyboardInterrupt:
        pass
    except IOError as e:  # catch closing of pipes
        if e.errno != errno.EPIPE:
            raise e
