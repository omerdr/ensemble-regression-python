from __future__ import print_function

from keras.layers import Dense
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.base import BaseEstimator, RegressorMixin

# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import mean_squared_error
# from regression_datasets import DatasetFactory
# from matplotlib.pyplot import plot, show
# import numpy as np


class MLPRegressor(BaseEstimator, RegressorMixin):
    """ Multi-Layer Perceptron implemented with Keras """
    def __init__(self, num_hidden_units=10, batch_size=32,nb_epoch=10):
        self.num_hidden_units = num_hidden_units
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

        self.fitted = False
        self.input_size = None
        self.model = Sequential()
        self.scaler = preprocessing.StandardScaler()
        pass

    def fit(self, X, y):
        if self.input_size is not None and self.input_size != X.shape[1]:
            raise Exception(message='Trying to re-fit the model with different input shape')
        if not self.fitted:
            input_size = X.shape[1]
            self.model.add(Dense(output_dim=self.num_hidden_units, input_dim=input_size, activation='tanh'))
            self.model.add(Dense(1))
            self.model.compile('sgd','mse')

        self.scaler.fit(X)  # standardize values of X so that every feature ~N(0,1)
        history = self.model.fit(self.scaler.transform(X), y, verbose=0,
                                 batch_size=self.batch_size,nb_epoch=self.nb_epoch)
        self.fitted = True
        return history

    def predict(self, X):
        if self.fitted:
            return self.model.predict(self.scaler.transform(X), verbose=0).squeeze()
        else:
            raise Exception('fit needs to be called before predict')


# dataset = DatasetFactory.boston()
# Xtrain,Xtest,ytrain,ytest = train_test_split(dataset.data, dataset.target)
# regr = MLPRegressor(10)
# h=regr.fit(Xtrain, ytrain)
# ypred = regr.predict(Xtest)
# regr.model.summary()
# np.set_printoptions(precision=3)
# print(regr.model.get_weights())
# print('MSE = ' + str(mean_squared_error(ytest,ypred)))
# plot(h.history['loss'])
# show()

