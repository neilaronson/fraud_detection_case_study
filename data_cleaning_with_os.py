import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from itertools import izip

class DataCleaning(object):
    """An object to clean and wrangle data into format for a model"""

    def __init__(self, filepath, training=True):
        """Reads in data

        Args:
            fileapth (str): location of file with csv data
        """
        self.df = pd.read_json(filepath)
        self.df['fraud'] = self.df['acct_type'].isin(['fraudster_event', 'fraudster', 'fraudster_att'])
        Y = self.df.pop('fraud').values
        X = self.df.values
        cols = self.df.columns
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=.8, random_state=123)
        self.training = pd.DataFrame(X_train, columns=cols)
        self.y_train = self.training['fraud']
        self.test = pd.DataFrame(X_test, columns=cols)
        self.y_test = self.test['fraud']
        if training:
            self.df = self.training
        else:
            print "using test data"
            self.df = self.test


    def make_target_variable(self):
        """Create the churn column, which is our target variable y that we are trying to predict
        A customer is considered churned if they haven't taken trip in last 30 days"""
        self.df['last_trip_date'] = pd.to_datetime(self.df['last_trip_date'])
        self.df['Churn'] = (self.df.last_trip_date < '2014-06-01').astype(int)

    def dummify(self, columns):
        """Create dummy columns for categorical variables"""
        dummies = pd.get_dummies(self.df[columns], prefix=columns)
        self.df = self.df.drop(columns, axis=1)
        self.df = pd.concat([self.df,dummies], axis=1)

    def drop_date_columns(self):
        """Remove date columns from feature matrix, avoid leakage"""
        self.df.drop('last_trip_date', axis=1, inplace=True)
        self.df.drop('signup_date', axis=1, inplace=True)

    def get_column_names(self):
        """Get the names of columns currently in the dataframe"""
        return list(self.df.columns.values)

    def cut_outliers(self, col):
        """Remove numerical data more than 3x outside of a column's std"""
        std = self.df[col].std()
        t_min = self.df[col].mean() - 3*std
        t_max = self.df[col].mean() + 3*std
        self.df = self.df[(self.df[col] >= t_min) & (self.df[col] <= t_max)]

    def drop_na(self):
        """Generic method to drop all rows with NA's in any column"""
        self.df = self.df.dropna(axis=0, how='any')

    def make_log_no_trips(self):
        """Transform the number of trips column to log scale"""
        self.df['log_trips'] = self.df[(self.df['trips_in_first_30_days'] != 0)].trips_in_first_30_days.apply(np.log)
        self.df['log_trips'] = self.df['log_trips'].apply(lambda x: 0 if np.isnan(x) else x)

    def drop_columns_for_regression(self):
        """Drop one of the dummy columns for when using a regression model"""
        self.df = self.df.drop(['phone_iPhone', 'city_Astapor'], axis=1)

    def mark_missing(self, cols):
        """Fills in NA values for a column with the word "missing" so that they won't be dropped later on"""
        for col in cols:
            self.df[col].fillna('missing', inplace=True)

    def drop_all_non_numeric(self):
        self.df = self.df.head(1000)
        self.df = self.df[['fraud', 'body_length', 'channels', 'num_payouts', 'org_twitter']]
        #self.df.drop(['approx_payout_date', 'country',  ])

    def div_count_pos_neg(self, X, y):
        """Helper function to divide X & y into positive and negative classes
        and counts up the number in each.

        Parameters
        ----------
        X : ndarray - 2D
        y : ndarray - 1D

        Returns
        -------
        negative_count : Int
        positive_count : Int
        X_positives    : ndarray - 2D
        X_negatives    : ndarray - 2D
        y_positives    : ndarray - 1D
        y_negatives    : ndarray - 1D
        """
        negatives, positives = y == 0, y == 1
        negative_count, positive_count = np.sum(negatives), np.sum(positives)
        X_positives, y_positives = X[positives], y[positives]
        X_negatives, y_negatives = X[negatives], y[negatives]
        return negative_count, positive_count, X_positives, \
               X_negatives, y_positives, y_negatives

    def oversample(self, X, y, tp):
       """Randomly choose positive observations from X & y, with replacement
       to achieve the target proportion of positive to negative observations.

       Parameters
       ----------
       X  : ndarray - 2D
       y  : ndarray - 1D
       tp : float - range [0, 1], target proportion of positive class observations

       Returns
       -------
       X_undersampled : ndarray - 2D
       y_undersampled : ndarray - 1D
       """
       if tp < np.mean(y):
           return X, y
       neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = self.div_count_pos_neg(X, y)
       positive_range = np.arange(pos_count)
       positive_size = (tp * neg_count) / (1 - tp)
       positive_idxs = np.random.choice(a=positive_range,
                                        size=int(positive_size),
                                        replace=True)
       X_positive_oversampled = X_pos[positive_idxs]
       y_positive_oversampled = y_pos[positive_idxs]
       X_oversampled = np.vstack((X_positive_oversampled, X_neg))
       y_oversampled = np.concatenate((y_positive_oversampled, y_neg))

       return X_oversampled, y_oversampled

    def clean(self, regression=False):
        """Executes all cleaning methods in proper order. If regression, remove one
        dummy column and scale numeric columns for regularization"""
        self.drop_all_non_numeric()
        self.drop_na()

        y = self.df.pop('fraud').values

        if regression:
            self.drop_columns_for_regression()
            for col in ['avg_dist', 'avg_rating_by_driver', 'avg_surge', 'surge_pct', 'trips_in_first_30_days', 'weekday_pct']:
                self.df[col] = scale(self.df[col])

        X = self.df.values

        X_oversampled, y_oversampled = self.oversample(X, y, tp=0.3)

        return X_oversampled, y_oversampled
