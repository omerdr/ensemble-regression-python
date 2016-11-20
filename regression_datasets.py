"""
    Exports regression datasets.
    Each dataset has object has dataset.data containing the covariats (x's),
    and dataset.target containing the response (y).
"""
import calendar
import os
from datetime import datetime

import numpy as np
from matplotlib.dates import datestr2num
from sklearn import datasets
import statsmodels.api as sm

UCI_DATASETS_BASE_DIR = r'/home/omer/Downloads/Datasets/UCI Datasets/'
SKLEARN_DATASETS_DIR = r'~/Downloads/Datasets/sklearn/'
FINANCE_DATASETS_DIR = r'/home/omer/Downloads/Datasets/misc/finance'
TITANIC_DATASET_FILE = r'/home/omer/Downloads/Datasets/misc/titanic/titanic_filtered_missing_ages.tsv'
FLIGHTS_DATASET_FILE = r'/home/omer/Downloads/Datasets/misc/flights/2008-preprocessed.csv'
BLOCKBUSTER_DATASET_FILE = r'/home/omer/Downloads/Datasets/misc/blockbuster-ratings/blockbuster-preprocessed.csv'
BASKETBALL_DATASET_FILE = r'/home/omer/Downloads/Datasets/misc/basketball/basketball-preprocessed-50k.tsv'

class String2UID:
    def __init__(self):
        self._mapping = dict()
        self._last = 1

    def __call__(self, *args):
        if len(args) != 1:
            raise ValueError
        if args[0] not in self._mapping.keys():
            self._mapping[args[0]] = self._last
            self._last += 1

        return self._mapping[args[0]]

    def reverse_lookup(self, id):
        for k, v in self._mapping.iteritems():
            if v == id:
                return k
        raise LookupError


def dateconv(s):
    """ date conversion helper. Format %Y-%m-%d """
    return calendar.timegm(datetime.strptime(s,'%Y-%m-%d').timetuple())


# dict for all methods that generate datasets
dataset_list = {}
def dataset_generator(func):
    dataset_list[func.__name__] = func
    return func


class DatasetFactory:
    class Dataset:
        def __init__(self, data=None, target=None, inner_object=None, sklearn_dataset=None):
            if not sklearn_dataset:
                self.data = data
                self.target = target
                self.inner_object = inner_object
            else:
                self.data = sklearn_dataset.data
                self.target = sklearn_dataset.target
                self.inner_object = sklearn_dataset

    @staticmethod
    @dataset_generator
    def boston():
        """ sklearn Boston housing dataset """
        return DatasetFactory.Dataset(sklearn_dataset=datasets.load_boston())

    @staticmethod
    @dataset_generator
    def diabetes():
        """ sklearn diabetes dataset """
        return DatasetFactory.Dataset(sklearn_dataset=datasets.load_diabetes())

    @staticmethod
    @dataset_generator
    def abalone():
        """ abalone age dataset """
        d = datasets.fetch_mldata('regression-datasets abalone',
                                  data_home=SKLEARN_DATASETS_DIR)
        return DatasetFactory.Dataset(data=d.data, target=d.int1[0,].astype(float), inner_object=d)

    @staticmethod
    @dataset_generator
    def friedman1(n_samples=20000):
        """ Generated data """
        (data, target) = datasets.make_friedman1(n_samples=n_samples)
        return DatasetFactory.Dataset(data=data, target=target)

    @staticmethod
    @dataset_generator
    def friedman2(n_samples=20000):
        """ Generated data """
        (data, target) = datasets.make_friedman2(n_samples=n_samples)
        return DatasetFactory.Dataset(data=data, target=target)

    @staticmethod
    @dataset_generator
    def friedman3(n_samples=20000):
        """ Generated data """
        (data, target) = datasets.make_friedman3(n_samples=n_samples)
        return DatasetFactory.Dataset(data=data, target=target)

    @staticmethod
    @dataset_generator
    def blog_feedback():
        """
        Instances in this dataset contain features extracted from blog posts.
        The task associated with the data is to predict how many comments the post will receive.
        """
        csvdata = np.loadtxt(os.path.join(UCI_DATASETS_BASE_DIR, 'BlogFeedback', 'blogData_train.csv'),
                             delimiter=',')
        (data, target) = csvdata[:, 0:-1], csvdata[:, -1]
        return DatasetFactory.Dataset(data=data, target=target)

    @staticmethod
    @dataset_generator
    def ccpp():
        """ Engine power consumption dataset """
        csvdata = np.loadtxt(os.path.join(UCI_DATASETS_BASE_DIR, 'CCPP', 'Folds5x2_pp.csv'),
                             delimiter=',', skiprows=1)
        (data, target) = csvdata[:, 0:-1], csvdata[:, -1]
        return DatasetFactory.Dataset(data=data, target=target)

    @staticmethod
    @dataset_generator
    def wine_quality_white():
        """ Wine quality dataset, data for white wines """
        csvdata = np.loadtxt(os.path.join(UCI_DATASETS_BASE_DIR, 'wine-quality', 'winequality-white.csv'),
                             delimiter=';', skiprows=1)
        (data, target) = csvdata[:, 0:-1], csvdata[:, -1]
        return DatasetFactory.Dataset(data=data, target=target)

    @staticmethod
    @dataset_generator
    def bike_sharing():
        """ Dataset about bike sharing service statistics """
        csvdata = np.loadtxt(os.path.join(UCI_DATASETS_BASE_DIR, 'Bike-Sharing', 'hour.csv'),
                             delimiter=',', skiprows=1, converters={1: datestr2num})
        (data, target) = csvdata[:, 0:-1], csvdata[:, -1]
        return DatasetFactory.Dataset(data=data, target=target)

    @staticmethod
    @dataset_generator
    def online_videos():
        """ YouTube video transcoding dataset """
        uid1 = String2UID()
        uid2 = String2UID()
        uid3 = String2UID()
        csvdata = np.loadtxt(os.path.join(UCI_DATASETS_BASE_DIR, 'online_video_dataset', 'transcoding_mesurment.tsv'),
                             delimiter='\t', skiprows=1, converters={0: uid1, 2: uid2, 15: uid3})
        (data, target) = csvdata[:, 0:-1], csvdata[:, -1]
        return DatasetFactory.Dataset(data=data, target=target)

    @staticmethod
    @dataset_generator
    def ratings_of_sweets():
        """ Ratings of sweets for collaborative-filtering. Data gathered on http://sweetrs.org/ website. """
        d = datasets.fetch_mldata('ratings-of-sweets-sweetrs',
                                  data_home=SKLEARN_DATASETS_DIR)
        return DatasetFactory.Dataset(data=d.data[:, 0:-1], target=d.data[:, -1], inner_object=d)

    @staticmethod
    @dataset_generator
    def affairs():
        """
        Extramarital affair data used to explain the allocation of an individual's time among work,
        time spent with a spouse, and time spent with a paramour.
        The data is used as an example of regression with censored data
        http://statsmodels.sourceforge.net/0.6.0/datasets/generated/fair.html
        """
        d = sm.datasets.fair.load()
        return DatasetFactory.Dataset(data=d.exog[:, 0:-1], target=d.exog[:, -1], inner_object=d)

    @staticmethod
    @dataset_generator
    def SP500():
        """
        Historical Data for S&P 500 Stocks
        A file in Historical Data Format contains one record per line of text corresponding to the data for
        a single date. The record is arranged into fields representing respectively the
        Ticker, Date, Open, High, Low, Close, Volume for the day.
        The fields are delimited by commas, and the records are delimited by <cr><lf>
        http://pages.swcp.com/stocks/#historical%20data
        """
        uid = String2UID()
        csvdata = np.loadtxt(os.path.join(FINANCE_DATASETS_DIR, r'S&P500_from_swcp', 'sp500hst_all.csv'),
                             delimiter=',', converters={0: uid})

        (data, target) = csvdata[:, 0:-1], csvdata[:, -1]

        # return uid as inner_object so that the user can go back from stock index to the stock
        return DatasetFactory.Dataset(data=data, target=target, inner_object=uid)

    @staticmethod
    @dataset_generator
    def nasdaq_index():
        """
        Historical Data for NASDAQ Index
        """
        uid = String2UID()
        csvdata = np.loadtxt(os.path.join(FINANCE_DATASETS_DIR, r'NASDAQ_from_my_github_stocks', 'NASDAQ-index.csv'),
                             delimiter=',', converters={0: dateconv}, skiprows=1)

        (data, target) = csvdata[:, 0:-1], csvdata[:, -1]

        # return uid as inner_object so that the user can go back from stock index to the stock
        return DatasetFactory.Dataset(data=data, target=target, inner_object=uid)

    @staticmethod
    def nasdaq_stocks(tick):
        """
        Historical quotes for NASDAQ-traded companies
        :param tick: ticker label (e.g. 'GOOGL', 'AAPL', etc.)
        """
        csvdata = np.loadtxt(
                os.path.join(FINANCE_DATASETS_DIR, r'NASDAQ_from_my_github_stocks','nasdaq_split_mat', tick + '.csv'),
                skiprows=1, delimiter=',')

        (data, target) = csvdata[:, 0:-1], csvdata[:, -1]
        return DatasetFactory.Dataset(data=data, target=target)

    @staticmethod
    @dataset_generator
    def nasdaq_GOOGL():
        return DatasetFactory.nasdaq_stocks('GOOGL')

    @staticmethod
    @dataset_generator
    def nasdaq_AAPL():
        return DatasetFactory.nasdaq_stocks('AAPL')

    @staticmethod
    @dataset_generator
    def nasdaq_NFLX():
        return DatasetFactory.nasdaq_stocks('NFLX')

    @staticmethod
    @dataset_generator
    def titanic():
        """
        Titanic dataset contains the following information:
        0:PassengerId, 1:Survived, 2:Pclass, 3:Name, 4:Sex, 5:Age, 6:SibSp, 7:Parch, 8:Ticket, 9:Fare, 10:Cabin, 11:Embarked

        The regression task that this function sets up is trying to predict a passenger's age according to the other covariats
        """

        sex_uid = String2UID()
        cabin_uid = String2UID()
        embarked_uid = String2UID()
        csvdata = np.loadtxt(TITANIC_DATASET_FILE, skiprows=1, delimiter='\t',
            converters={4: sex_uid, 10:cabin_uid, 11:embarked_uid},
            usecols=(0,1,2,4,5,6,7,9,11))  # not using name, cabin and ticket

        cols = range(csvdata.shape[1])
        cols.remove(4)
        (data, target) = csvdata[:, cols], csvdata[:, 4]  # age is now the 4th field (since name was ignored)

        # return uid as inner_object so that the user can go back from stock index to the stock
        return DatasetFactory.Dataset(data=data, target=target)

    @staticmethod
    def flights(origin_airport=None):
        """
        Information on flights from 2008, with the task of trying to predict the delay upon arrival in minutes.
        Date,DayOfWeek,DepTime,ScheduledDepTime,ScheduledArrTime,FlightID,TailNum,Origin,Dest,Distance,ArrDelay
        2008-1-3,4,2003,1955,2225,WN335,N712SW,IAD,TPA,810,-14
        http://stat-computing.org/dataexpo/2009/the-data.html
        :param origin_airport: ATL,JFK,SFO,BOS for only using data from that specific airport. None for all origins.
        """
        def localminute(s):  # helper to convert HHMM to number (minute number in the day - 0-24*60).
            if s == '2400':
                return 24*60
            d = datetime.strptime(s.zfill(4),'%H%M')  # zfill adds missing 0's
            return d.minute + 60*d.hour

        flight_id = String2UID()  # carrier + flight number (e.g. "BA123" for British-Airways flight 123)
        tail_id = String2UID()
        location_id = String2UID() # same id for origin and destination

        filename = FLIGHTS_DATASET_FILE
        if origin_airport:
            filename = ('-'+origin_airport).join(os.path.splitext(filename))
        csvdata = np.loadtxt(filename, skiprows=1, delimiter=',',
                             converters={0: dateconv, 2:localminute, 3:localminute, 4:localminute, 5: flight_id,
                                         6: tail_id, 7:location_id, 8:location_id})

        (data, target) = csvdata[:, 0:-1], csvdata[:, -1]
        return DatasetFactory.Dataset(data=data, target=target)

    @staticmethod
    @dataset_generator
    def flights_JFK():
        return DatasetFactory.flights('JFK')

    @staticmethod
    @dataset_generator
    def flights_BOS():
        return DatasetFactory.flights('BOS')

    @staticmethod
    @dataset_generator
    def flights_BWI():
        return DatasetFactory.flights('BWI')

    @staticmethod
    @dataset_generator
    def flights_AUS():
        return DatasetFactory.flights('AUS')

    @staticmethod
    @dataset_generator
    def flights_HOU():
        return DatasetFactory.flights('HOU')

    @staticmethod
    @dataset_generator
    def flights_LGA():
        return DatasetFactory.flights('LGA')

    @staticmethod
    @dataset_generator
    def flights_longhaul():
        return DatasetFactory.flights('longhaul')

    @staticmethod
    @dataset_generator
    def blockbuster():
        """
        Dataset contains the genres (3), imdb rating and rotten tomato rating.
        Goal is to predict what the movie grossed in cinemas.
        """
        genre_id = String2UID()
        csvdata = np.loadtxt(BLOCKBUSTER_DATASET_FILE, skiprows=1, delimiter=',',
                             converters={0: genre_id, 1:genre_id, 2:genre_id})

        (data, target) = csvdata[:, 0:-1], csvdata[:, -1]
        return DatasetFactory.Dataset(data=data, target=target)

    @staticmethod
    @dataset_generator
    def basketball():
        """
        Dataset contains stats on NBA players. Task: Predict points scored on the next game.
        0:name,1:venue,2:team,3:date,4:start,5:pts_ma,6:min_ma,7:pts_ma_1,8:min_ma_1,9:pts
        explanation:
            start is whether or not the player started.
            pts is number of points scored, min is number of minutes played.
            ma stands for moving average, starts at season.
            ma_1 is moving average with a 1 game lag.
        """
        name = String2UID()
        venue = String2UID()
        team = String2UID()
        start = String2UID()
        csvdata = np.loadtxt(BASKETBALL_DATASET_FILE, skiprows=1, delimiter='\t',
                             converters={0: name, 1:venue, 2:team, 3:dateconv, 4:start})

        (data, target) = csvdata[:, 0:-1], csvdata[:, -1]
        return DatasetFactory.Dataset(data=data, target=target)

# idxs = (dataset.target > 435) & (dataset.target < 450)
# dataset.target = dataset.target[idxs]
# dataset.data = dataset.data[idxs,:]
