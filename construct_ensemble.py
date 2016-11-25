from __future__ import print_function

import sys
import time
import errno
import traceback
from os import path
from threading import Lock

import numpy as np
import pandas as pd
import scipy.io as sio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dummy_pool_executor import DummyPoolExecutor
from sklearn import preprocessing, model_selection, linear_model

from ensemble_regressor import EnsembleRegressor
from plotting_tools import plot_regression_results, plot_y_e_correlation
from regression_datasets import DatasetFactory, dataset_list


def make_large_ensemble(dataset, mat_filename='large_ensemble.mat'):
    """
    construct_ensemble splits the dataset into train and 'test'. The ensemble regressors are trained on the training
    set. The test set is saved to the mat file to be used by matlab code.

    :param dataset: a dataset object created by DatasetFactory
    :param mat_filename: name of the mat file to save the results to
    # :param ensemble_type: 'auto' or 'mlp' for the choice of regressors in the ensemble
    # :param train_size: proportion or number of samples used for training (defaults to 25% for n>20,000, otherwise 50%)
    # :param test_size: proportion or number of samples used for testing (defaults to 75% for n>20,000, and 50% otherwise)
    # :param samples_per_regressor: Number of samples from X_train that each regressor will be trained on.
    #                               Default 'None' will cause all regressors to be trained on all samples.
    # :param overlap: this is the number of samples overlapping for every adjacent pair of regressors.
    #        Defaults to no overlap if there are at least 100 samples per regressor, else full overlap (overlap=n).
    # :param plotting: plots results
    # :param ensemble_train_size: The number of samples to output for training supervised ensemble methods
    # :param scale_data: boolean, if True the data will be scaled to mean-centered and variance 1.
    # :param Description: The text that will be written in the 'Description' field in the output file
    """

    # Init
    ensemble_type = 'mlp_large'  # 'auto_large'  ######################################################################
    (n_samples,n_features) = dataset.data.shape

    ensemble = EnsembleRegressor(verbose=False, type=ensemble_type)

    m = ensemble.regressor_count
    train_size = np.max([200, 100*dataset.data.shape[1]])  # at least 100 samples per dimension (n/p >= 100)
    n = n_samples - train_size
    n_train = 200  # taken out of val_size
    samples_per_regressor = train_size#//m
    overlap = samples_per_regressor  # 0

    # scale data
    dataset.data = preprocessing.scale(dataset.data)

    # split to train / validation
    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        dataset.data, dataset.target, random_state=0,
        train_size=train_size, test_size=n)

    msg = "features=%s, n_tot=%d, training each regressors on %d, n=%d, n_train=%d, m=%d" % \
          (n_features, n_samples, samples_per_regressor, n, n_train, m)
    print(msg)

    ensemble.fit(X_train, y_train) # full overlap #, samples_per_regressor=samples_per_regressor, regressor_overlap=overlap)

    scores_train = ensemble.score(X_train, y_train)
    MSEs_train = ensemble.mean_squared_error(X_train, y_train) / np.var(y_train)
    scores_val = ensemble.score(X_val, y_val)
    MSEs_val = ensemble.mean_squared_error(X_val, y_val) / np.var(y_val)

    for i, regr in enumerate(ensemble.regressors):
        print('## ' + str(i) + '. ' + regr.__class__.__name__ + ':')
        print(regr)

        print('\tMSE/Var(Y): %.2f/%.2f' % (MSEs_train[i],MSEs_val[i]))
        print('\tVariance score (R^2): %.2f/%.2f\n' % (scores_train[i],scores_val[i]))

    # create predictions matrix on the test set
    Zval = ensemble.predict(X_val)

    # Set aside n_train samples as a training set for the supervised ensemble learners
    Z_train, Z, y_ensemble_train, y_ensemble_test = \
        model_selection.train_test_split(Zval.T, y_val, random_state=42, train_size=n_train)
    Z_train = Z_train.T
    Z = Z.T

    # Add Description if none
    Description = "%s was generated with %s regressors of type %s:\n%s" % \
                  (mat_filename, msg, ensemble_type, str(locals()))

    sio.savemat(mat_filename, {
        'names': ensemble.regressor_labels,
        'Z': Z, 'y': y_ensemble_test,
        'Ztrain': Z_train, 'ytrain': y_ensemble_train,
        'samples_per_regressor': samples_per_regressor,
        'regressor_samples_overlap': overlap,
        'Ey': np.mean(y_ensemble_test),  # np.mean(dataset.target),
        'Ey2': np.mean(y_ensemble_test ** 2),  # np.mean(dataset.target ** 2)
        'Description': Description
    })

    results_df = pd.DataFrame(
        {'i': range(1, 1+len(MSEs_train)), 'MSE_train': MSEs_train, 'MSE_val': MSEs_val,
                            'R2_train': scores_train,'R2_val': scores_val})
    return results_df


def make_ensemble(dataset, mat_filename='ensemble.mat', ensemble_type='auto',
                  train_size=None, test_size=None, samples_per_regressor=None, overlap=None, plotting=True,
                  Description=None, scale_data=False, ensemble_train_size=200):
    """
    construct_ensemble splits the dataset into train and 'test'. The ensemble regressors are trained on the training
    set. The test set is saved to the mat file to be used by matlab code.

    :param dataset: a dataset object created by DatasetFactory
    :param mat_filename: name of the mat file to save the results to
    :param ensemble_type: 'auto' or 'mlp' for the choice of regressors in the ensemble
    :param train_size: proportion or number of samples used for training (defaults to 25% for n>20,000, otherwise 50%)
    :param test_size: proportion or number of samples used for testing (defaults to 75% for n>20,000, and 50% otherwise)
    :param samples_per_regressor: Number of samples from X_train that each regressor will be trained on.
                                  Default 'None' will cause all regressors to be trained on all samples.
    :param overlap: this is the number of samples overlapping for every adjacent pair of regressors.
           Defaults to no overlap if there are at least 100 samples per regressor, else full overlap (overlap=n).
    :param plotting: plots results
    :param ensemble_train_size: The number of samples to output for training supervised ensemble methods
    :param scale_data: boolean, if True the data will be scaled to mean-centered and variance 1.
    :param Description: The text that will be written in the 'Description' field in the output file
    """
    if scale_data:
        dataset.data = preprocessing.scale(dataset.data)

    if (train_size is None) and (test_size is None):
        if len(dataset.target) < 20000:
            (test_size,train_size) = (0.75, 0.25)
        else:
            (test_size, train_size) = (0.5, 0.5)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        dataset.data,  # preprocessing.scale(dataset.data)
        dataset.target, random_state=0, test_size=test_size, train_size=train_size)

    # Prepare ensemble regressors
    ensemble = EnsembleRegressor(verbose=False, type=ensemble_type)

    n = len(y_train)
    m = ensemble.regressor_count

    # decide on how many samples per regressor and what's the overlap between regressors
    if samples_per_regressor and (overlap is not None):
        pass # both were defined by caller

    elif (overlap is not None) and (samples_per_regressor is None):
        samples_per_regressor = (n // m) + overlap  # '//' is python operator for floor of n/m

    else: # both are None or only samples_per_regressor was given
        if n < m*100:  # reserve at least 100 samples for training the individual regressors
            overlap = n
            samples_per_regressor = (samples_per_regressor or n)
        else:  # we have enough samples to be training on different parts of the dataset
            overlap = 0
            samples_per_regressor = (samples_per_regressor or n // m)

    assert train_size == (samples_per_regressor*m) - overlap*(m-1), "inconsistent parameters"


    print("Training set size: %d with %d attributes" % X_train.shape)
    print("Each regressor is trained on %d samples" % samples_per_regressor)
    print("Test set size: %d" % len(y_test))

    ensemble.fit(X_train, y_train, samples_per_regressor=samples_per_regressor, regressor_overlap=overlap)
    scores = ensemble.score(X_train, y_train)
    MSEs = ensemble.mean_squared_error(X_train, y_train)

    for i, regr in enumerate(ensemble.regressors):
        print('## ' + str(i) + '. ' + regr.__class__.__name__ + ':')
        print(regr)

        print('\tMSE: %.2f' % MSEs[i])
        print('\tVariance score (R^2): %.2f\n' % scores[i])

    # create predictions matrix on the test set
    Z = ensemble.predict(X_test)

    # Set aside 200 samples as a training set for the supervised ensemble learners
    Z_train, Z, y_ensemble_train, y_ensemble_test = \
        model_selection.train_test_split(Z.T, y_test, random_state=0, train_size=ensemble_train_size)
    Z_train = Z_train.T
    Z = Z.T

    # Add Description if none
    if not Description:
        Description = "%s was generated with %d samples and %d regressors of type %s:\n%s" % \
                      (mat_filename, n, m, ensemble_type, str(locals()))

    sio.savemat(mat_filename, {
        'names': ensemble.regressor_labels,
        'Z': Z, 'y': y_ensemble_test,
        'Ztrain': Z_train, 'ytrain': y_ensemble_train,
        'samples_per_regressor': samples_per_regressor,
        'regressor_samples_overlap': overlap,
        'Ey': np.mean(y_ensemble_test),  # np.mean(dataset.target),
        'Ey2': np.mean(y_ensemble_test ** 2),  # np.mean(dataset.target ** 2)
        'Description': Description
    })

    if plotting:
        plot_regression_results(ensemble, Z, y_ensemble_test)
        plot_y_e_correlation(ensemble, Z, y_ensemble_test)

# region <SpecialTestCases>
def RidgeRegressionEnsembleTest():
    #dataset = DatasetFactory.friedman1(n_samples=200200)
    #dataset = DatasetFactory.friedman2(n_samples=200200)
    dataset = DatasetFactory.friedman3(n_samples=200200)
    Xtrain, X, ytrain, y = model_selection.train_test_split(
        dataset.data,  dataset.target, random_state=0, train_size=200)
    ensemble = EnsembleRegressor(type='ridge')
    ensemble.fit(Xtrain,ytrain,samples_per_regressor=200,regressor_overlap=200)
    ridgecv = linear_model.RidgeCV(alphas=np.arange(.1,1,.2), fit_intercept=True, normalize=True)
    ridgecv.fit(Xtrain,ytrain)
    y_ridgecv = ridgecv.predict(X)
    Z = ensemble.predict(X)

    sio.savemat('RidgeRegression_Friedman3_200k.mat', {
        'names': ensemble.regressor_labels,
        'Z': Z, 'y': y,
        # 'Ztrain': Z_train, 'ytrain': ytrain,
        'y_RidgeCV': y_ridgecv,
        'samples_per_regressor': 200,
        'regressor_samples_overlap': 200,
        'Ey': np.mean(y),
        'Ey2': np.mean(y ** 2),
        'Description': 'Ridge Regression (Friedman #3)'
    })

def DifferentRegressorsEnsembleTest():
    dataset = DatasetFactory.friedman1(n_samples=200200)
    #dataset = DatasetFactory.friedman2(n_samples=200200)
    #dataset = DatasetFactory.friedman3(n_samples=200200)
    Xtrain, X, ytrain, y = model_selection.train_test_split(
        dataset.data,  dataset.target, random_state=0, train_size=200)
    ensemble = EnsembleRegressor(type='auto_large')
    ensemble.fit(Xtrain,ytrain,samples_per_regressor=200,regressor_overlap=200)
    Ztrain = ensemble.predict(Xtrain)
    Z = ensemble.predict(X)

    sio.savemat('DifferentRegressors_Friedman1.mat', {
        'names': ensemble.regressor_labels,
        'Z': Z, 'y': y,
        'Ztrain': Ztrain, 'ytrain': ytrain,
        'samples_per_regressor': 200,
        'regressor_samples_overlap': 200,
        'Ey': np.mean(y),
        'Ey2': np.mean(y ** 2),
        'Description': 'Different Regressors (Friedman #1)'
    })

def UnequalMLPsEnsembleTest():
    #dataset = DatasetFactory.friedman1(n_samples=200200)
    #dataset = DatasetFactory.friedman2(n_samples=200200)
    dataset = DatasetFactory.friedman3(n_samples=200200)
    Xtrain, X, ytrain, y = model_selection.train_test_split(
        dataset.data,  dataset.target, random_state=0, train_size=200)
    ensemble = EnsembleRegressor(type='auto_large')
    ensemble.fit(Xtrain,ytrain,samples_per_regressor=200,regressor_overlap=200)
    Ztrain = ensemble.predict(Xtrain)
    Z = ensemble.predict(X)

    sio.savemat('ManualEnsembleDatasets\DifferentRegressors_Friedman3.mat', {
        'names': ensemble.regressor_labels,
        'Z': Z, 'y': y,
        'Ztrain': Ztrain, 'ytrain': ytrain,
        'samples_per_regressor': 200,
        'regressor_samples_overlap': 200,
        'Ey': np.mean(y),
        'Ey2': np.mean(y ** 2),
        'Description': 'Different Regressors (Friedman #3)'
    })


def RealDatasetsManualEnsembleTest():
    for name,func in dataset_list.iteritems():
        print(name + ":", end="")
        dataset = func()
        print(" X.shape = " + str(dataset.data.shape))
        ensemble = EnsembleRegressor(type='auto', verbose=True)  #auto_large

        if name is 'blog_feedback':
            continue
            # samples_per_regressor = 2810
            # overlap = 2810
            # train_size = 2810
        else:
            samples_per_regressor = 200
            overlap = 0
            train_size = samples_per_regressor * ensemble.regressor_count

        if len(dataset.target) < train_size + 500:  # ignore datasets with less than 6000 samples
            continue
        # if dataset.data.shape[1] < 5:  # ignore datasets with less than 5 covariates
        #     continue

        Xtrain, X, ytrain, y = model_selection.train_test_split(
            dataset.data, dataset.target, random_state=0, train_size=train_size)

        ensemble.fit(Xtrain, ytrain, samples_per_regressor=samples_per_regressor, regressor_overlap=overlap)
        Ztrain = ensemble.predict(Xtrain)
        Z = ensemble.predict(X)

        sio.savemat(path.join('ManualEnsembleDatasets',name + '.mat'), {
            'names': ensemble.regressor_labels,
            'Z': Z, 'y': y,
            'Ztrain': Ztrain, 'ytrain': ytrain,
            'samples_per_regressor': train_size,
            'regressor_samples_overlap': train_size,
            'Ey': np.mean(y),
            'Ey2': np.mean(y ** 2),
            'Description': ('Different Regressors (%s)' % name)
        })

def RealDatasetsLargeMLPEnsembleTest():
    for name,func in dataset_list.iteritems():
        print(name)
        dataset = func()

        if len(dataset.target) < 5500:  # ignore datasets with less than 6000 samples
            continue
        if dataset.data.shape[1] < 5:  # ignore datasets with less than 5 covariates
            continue

        if name is 'blog_feedback':
            train_size = 10000
        else:
            train_size = 500

        Xtrain, X, ytrain, y = model_selection.train_test_split(
            dataset.data, dataset.target, random_state=0, train_size=train_size)

        if name is 'affairs':
            # ytrain, y = [np_utils.to_categorical(x) for x in (ytrain, y)]
            continue

        ensemble = EnsembleRegressor(type='mlp_large', verbose=True)
        ensemble.fit(Xtrain, ytrain, samples_per_regressor=train_size, regressor_overlap=train_size)
        Ztrain = ensemble.predict(Xtrain)
        Z = ensemble.predict(X)

        sio.savemat(path.join('ManualEnsembleDatasets',name + '_10mlp.mat'), {
            'names': ensemble.regressor_labels,
            'Z': Z, 'y': y,
            'Ztrain': Ztrain, 'ytrain': ytrain,
            'samples_per_regressor': train_size,
            'regressor_samples_overlap': train_size,
            'Ey': np.mean(y),
            'Ey2': np.mean(y ** 2),
            'Description': ('Different Regressors (%s)' % name)
        })

def RepeatRealDatasetsDifferentRegressorsTest():
    for name,func in dataset_list.iteritems():
        print(name)
        dataset = func()
        make_ensemble(dataset, "auto_repeat/auto_" + name + ".mat", plotting=False)

def RealDatasetsDifferentRegressorsLargeTest():
    for name,func in dataset_list.iteritems():
        print(name)
        dataset = func()
        make_ensemble(dataset, "auto_large/auto_" + name + ".mat", plotting=False,
                      ensemble_type='auto_large', scale_data=True)

# def MLPTestForGEM():
#     ensemble = EnsembleRegressor(type='mlp', verbose=True)
#     samples_per_regressor = 200
#     train_size = samples_per_regressor * ensemble.regressor_count
#     validation_size = 1000
#     dataset = DatasetFactory.friedman1(n_samples=train_size+validation_size)
#
#     Xtrain, X, ytrain, y = cross_validation.train_test_split(
#         dataset.data, dataset.target, random_state=0, train_size=train_size)
#
#     ensemble.fit(Xtrain, ytrain, samples_per_regressor=samples_per_regressor, regressor_overlap=0)
#     Ztrain = ensemble.predict(Xtrain)
#     Z = ensemble.predict(X)
#
#     sio.savemat(path.join('ManualEnsembleDatasets',name + '_10mlp.mat'), {
#         'names': ensemble.regressor_labels,
#         'Z': Z, 'y': y,
#         'Ztrain': Ztrain, 'ytrain': ytrain,
#         'samples_per_regressor': train_size,
#         'regressor_samples_overlap': train_size,
#         'Ey': np.mean(y),
#         'Ey2': np.mean(y ** 2),
#         'Description': ('Different Regressors (%s)' % name)
#     })

# endregion


def submit_one(data, target, filename):
    return make_large_ensemble(DatasetFactory.Dataset(data,target), filename)


def main():
    # RidgeRegressionEnsembleTest()
    # DifferentRegressorsEnsembleTest()
    # RealDatasetsManualEnsembleTest()
    # RealDatasetsLargeMLPEnsembleTest()
    # RepeatRealDatasetsDifferentRegressorsTest()
    # RealDatasetsDifferentRegressorsLargeTest()

    a = dict()
    # a['abalone'] = dataset_list['abalone']
    # a['wine_quality_white'] = dataset_list['wine_quality_white']
    # a['bike_sharing'] = dataset_list['bike_sharing']
    # a['ccpp'] = dataset_list['ccpp']
    # a['ratings_of_sweets'] = dataset_list['ratings_of_sweets']
    # a['affairs'] = dataset_list['affairs']
    # a['flights_BOS'] = dataset_list['flights_BOS']
    results_list = list()
    keys = list()
    with ProcessPoolExecutor() as pool:
        future_to_name_mapping = {}
        print('{0} datasets: {1}'.format(len(dataset_list), dataset_list.keys()))
        for name, func in dataset_list.items():  # a.items():  #
            print(name)
            dataset = func()
            if len(dataset.target) < 4000:
                continue

            try:
                ########################################################################################################
                # future = pool.submit(submit_one, dataset.data, dataset.target, 'final/misc/%s.mat' % name)
                future = pool.submit(submit_one, dataset.data, dataset.target, 'final/mlp/%s.mat' % name)
                future_to_name_mapping[future] = name
            except Exception as e:
                print('Exception while submitting {0}'.format(name), file=sys.stderr)
                traceback.print_tb(e.__traceback__)

        print('BEFORE')
        for future in as_completed(future_to_name_mapping):
            name = future_to_name_mapping[future]
            keys.append(name)
            print('{0} now completed. Already done {1}'.format(name, keys))
            try:
                res = future.result()
                cur_df = pd.DataFrame(res)
                results_list.append(cur_df)
                print(cur_df)
            except Exception as e:
                print('Exception caught while collecting results from {0}'.format(name), file=sys.stderr)
                traceback.print_tb(e.__traceback__)

        print('AFTER')
            # reg_results = pd.DataFrame(make_large_ensemble(dataset, 'final/misc/%s.mat' % name))
            # results_list.append(reg_results)
            # keys.append(name)
            # print(reg_results)

        results_df = pd.concat(results_list, keys=keys)
        results_df.to_csv('final/mlp/results.csv')  # ##################################################################
        pd.options.display.float_format = '{:.2f}'.format
        print(results_df.pivot_table(values=['MSE_train','MSE_val'], index=['i'], aggfunc=[np.mean,np.min,np.max]))
        print(results_df.reset_index(inplace=False).pivot_table(values=['MSE_train', 'MSE_val'], index='level_0',
                                                          aggfunc=[np.mean, np.min, np.max]))

    # region <Selected datasets>
    # make_ensemble(DatasetFactory.nasdaq_index(), "auto/auto_NASDAQ_index.mat")
    # make_ensemble(DatasetFactory.flights(origin_airport='longhaul'), "auto/auto_flights_longhaul.mat", plotting=False)
    # make_ensemble(DatasetFactory.blockbuster(), "auto/auto_blockbuster.mat")
    # make_ensemble(DatasetFactory.boston(), scale_data=True)
    # endregion


    # region <Semi-supervised experiments>
    # large friedman1
    # dataset = DatasetFactory.friedman1(10000+200+200*15)  # 10k unlabeled, 200 labeled (for supervised ensemble methods), 200 per regressor for training
    # make_ensemble(dataset, "auto_large/auto_large_friedman1.mat", plotting=False,
    #               ensemble_type='auto_large', scale_data=True, overlap=0, samples_per_regressor=200, train_size=200*15)

    # large nn friedman1
    # dataset = DatasetFactory.friedman1(10000 + 200 + 200 * 10)  # 10k unlabeled, 200 labeled (for supervised ensemble methods), 200 per regressor for training
    # make_ensemble(dataset, "auto_large/mlp_large_friedman1.mat", plotting=False,
    #               ensemble_type='mlp_large', scale_data=True, overlap=0, samples_per_regressor=200,
    #               train_size=200 * 10)

    # large nn friedman1 - more data
    # dataset = DatasetFactory.friedman1(50000 + 5000 + 200 * 10)  # 50k unlabeled, 5000 labeled (for supervised ensemble methods), 200 per regressor for training
    # make_ensemble(dataset, "auto_large/mlp_large_friedman1_big.mat", plotting=False,
    #               ensemble_type='mlp_large', scale_data=True, overlap=0, samples_per_regressor=200,
    #               train_size=200 * 10, ensemble_train_size=5000)

    # extra-large nn friedman1
    # dataset = DatasetFactory.friedman1(10000 + 200 + 200 * 30)  # 10k unlabeled, 200 labeled (for supervised ensemble methods), 200 per regressor for training
    # make_ensemble(dataset, "auto_large/mlp_xlarge_friedman1.mat", plotting=False,
    #               ensemble_type='mlp_xlarge', scale_data=True, overlap=0, samples_per_regressor=200,
    #               train_size=200 * 30)
    # endregion

    # region <All Datasets>
    # for name,func in dataset_list.iteritems():
    #     print(name)
    #     dataset = func()
    #     make_ensemble(dataset, "auto_mlp5_change_h_num/auto_" + name + ".mat", plotting=False)
    # endregion

    # region <Freidmans>
    # make_ensemble(DatasetFactory.friedman1(), "auto/auto_friedman1_new.mat")
    # make_ensemble(DatasetFactory.friedman2(), "auto/auto_friedman2_new.mat")
    # make_ensemble(DatasetFactory.friedman3(), "auto/auto_friedman3_new.mat")
    # endregion

    # for i in [6000, 1e4, 2e4, 5e4, 1e5, 5e5, 1e6]:
    #     for iter in range(10):
    #         make_ensemble(DatasetFactory.friedman1(n_samples=i),
    #                       mat_filename=("auto_friedman1_test3/auto_n_%d_iter#%d.mat" % (i,iter)),
    #                       ensemble_type='auto', train_size=1000, plotting=False, overlap=0)
    #         make_ensemble(DatasetFactory.friedman1(n_samples=i),
    #                       mat_filename=("auto_friedman1_test3/mlp_n_%d_iter#%d.mat" % (i, iter)),
    #                       ensemble_type='mlp', train_size=1000, plotting=False, overlap=0)

    # region <Friedman1 test data>
    # n_for_evaluation = 20200  # 1000 for training, 20000 for unsupervised algs
    # n_train_per_regressor = 200
    # #m = len(EnsembleRegressor._ensemble_regressors_auto) # 6
    # m = len(EnsembleRegressor._ensemble_nn) # 5
    # for overlap in range(0,101,10):
    #     n_for_training = (m * n_train_per_regressor) - (m-1) * overlap  # |-----%%-----%%-----%%-----|
    #     n = n_for_evaluation + n_for_training
    #     for iter in range(10):
    #         make_ensemble(DatasetFactory.friedman1(n_samples=n),
    #                       mat_filename=("auto_friedman1_test_corr/mlp_overlap_%d_iter_%d.mat" % (overlap, iter)),
    #                       ensemble_type='auto', train_size=n_for_training, plotting=False,
    #                       samples_per_regressor=n_train_per_regressor, overlap=overlap)
    # endregion


if __name__ == "__main__":
    try:
        start = time.time()
        main()
        print('Done.')
        print('Total running time: {:.2f}'.format(time.time() - start))
    except KeyboardInterrupt:
        pass
    except IOError as e:  # catch closing of pipes
        if e.errno != errno.EPIPE:
            raise e
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print('PROGRAM EXIT\nEXCEPTION RAISED: {0}'.format(e), file=sys.stderr)


