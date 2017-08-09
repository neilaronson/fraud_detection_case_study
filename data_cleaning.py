import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

class DataCleaning(object):
    """An object to clean and wrangle data into format for a model"""

    def __init__(self, filepath, training=True, predict=False):
        """Reads in data

        Args:
            fileapth (str): location of file with csv data
        """

        if not predict:
            self.df = pd.read_json(filepath)
            self.df['fraud'] = self.df['acct_type'].isin(['fraudster_event', 'fraudster', 'fraudster_att'])
            index_list = range(len(self.df))
            X_train, X_test = train_test_split(index_list, train_size=.8, random_state=123)
            training_data = self.df.iloc[X_train,:]
            test_data = self.df.iloc[X_test,:]

            if training:
                self.df = training_data
            else:
                print "using test data"
                self.df = test_data
        else: #predict=True
            one_row = pd.Series(filepath)
            self.df = pd.DataFrame([one_row])


    def dummify(self, columns):
        """Create dummy columns for categorical variables"""
        dummies = pd.get_dummies(self.df[columns], columns=columns,
                                prefix=columns)
        self.df = self.df.drop(columns, axis=1)
        self.df = pd.concat([self.df,dummies], axis=1)

    def get_column_names(self):
        """Get the names of columns currently in the dataframe"""
        return list(self.df.columns.values)

    def drop_na(self):
        """Generic method to drop all rows with NA's in any column. Currently not used"""
        self.df = self.df.dropna(axis=0, how='any')

    def drop_columns_for_regression(self):
        """Drop one of the dummy columns for when using a regression model"""
        self.df = self.df.drop(['has_header_0.0'], axis=1)

    def mark_missing(self, cols):
        """Fills in NA values for a column with the word "missing" so that they won't be dropped later on"""
        for col in cols:
            self.df[col].fillna('missing', inplace=True)

    def get_text(self, raw_html):
        soup = BeautifulSoup(raw_html, "html.parser")
        return soup.get_text()

    def add_plaintext(self):
        self.df['text_description'] = self.df['description'].apply(self.get_text)

    def assign_text_cluster(self):
        self.add_plaintext()

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

    def drop_some_cols(self, columns):
        """Simply drop columns (as list) from the dataframe"""
        for col in columns:
            self.df = self.df.drop(col,axis=1)

    def fix_listed(self):
        """Change the listed column from string to int (yes maps to 1, no maps to 0)"""
        self.df['listed'] = self.df['listed'].astype(str)
        d = {'y':1,'n':0}
        self.df['listed'] = self.df['listed'].map(d)

    def make_previous_payouts_total(self):
        """Add column for how many previous payouts the user received"""
        self.df['num_previous_payouts'] = self.df['previous_payouts'].apply(len)

    def make_total_tickets(self):
        """Add column total_tickets that shows the total quantity of tickets offered for event"""
        total_quantity = []
        for row in self.df['ticket_types']:
            quantity = 0
            for i in range(len(row)):
                quantity += row[i]['quantity_total']
            total_quantity.append(quantity)
        self.df['total_quantity'] = total_quantity

    def make_num_ticket_types(self):
        """Add column for the number of different ticket types"""
        self.df['num_ticket_types'] = self.df['ticket_types'].apply(len)

    def have_or_not(self,columns):
        """Fill in missing columns / whitespace with 'nan', then create new column to indicate if event has column value or not"""
        #org name, payee name
        for column in columns:
            self.df[column] = self.df[column].replace('',np.nan)
            new_col_name = str(column).replace(".", "_")
            self.df['has_'+new_col_name] = self.df[column].notnull().astype(int)

    def fix_have_header(self):
        """fix the has_header column by marking missing ones and dummifying"""
        self.mark_missing(['has_header'])
        self.dummify(['has_header'])

    def zero_versus_rest(self, columns):
        """for numeric columns, drop the columns and add 2 dummies: is zero or not 0"""
        for col in columns:
            self.df[col+"_is_0"] = (self.df[col]==0.0)*1
            #self.df[col+"_not_0"] = (self.df[col]!=0.0)*1
            self.df = self.df.drop(col, axis=1)

    def rename_columns(self):
        old_columns = self.df.columns
        new_columns = {old_name: old_name.replace(".", ",") for old_name in old_columns}
        self.df.rename(columns=new_columns, inplace=True)


    def clean(self, regression=False, predict=False, test=False):
        """Executes all cleaning methods in proper order. If regression, remove one
        dummy column and scale numeric columns for regularization"""
        self.fix_listed()
        self.make_previous_payouts_total()
        self.make_total_tickets()
        self.make_num_ticket_types()
        self.have_or_not(['org_name','payee_name', 'payout_type'])
        #self.fix_have_header()
        self.zero_versus_rest(['org_facebook', 'org_twitter', 'channels', 'delivery_method', 'num_order', 'num_payouts', 'user_age'])
        if predict:
            todrop = ['approx_payout_date', 'country', 'currency', 'description', 'email_domain', 'event_created', 'event_end', 'event_published', \
                        'event_start', 'fb_published', 'gts', 'has_analytics', 'has_logo', 'name', 'object_id', 'org_desc', 'org_name', \
                        'payee_name', 'payout_type', 'previous_payouts', 'sale_duration', 'sale_duration2', 'show_map', 'ticket_types', \
                        'user_created', 'user_type', 'venue_address', 'venue_country', 'venue_latitude', 'venue_longitude', 'venue_name', 'venue_state', 'has_header']
            # self.df['has_header_0.0'] = 0
            # self.df['has_header_1.0'] = 0
            # self.df['has_header_missing'] = 0
        else:
            todrop = ['acct_type', 'approx_payout_date', 'country', 'currency', 'description', 'email_domain', 'event_created', 'event_end', 'event_published', \
                        'has_header', 'event_start', 'fb_published', 'gts', 'has_analytics', 'has_logo', 'name', 'object_id', 'org_desc', 'org_name', \
                        'payee_name', 'payout_type', 'previous_payouts', 'sale_duration', 'sale_duration2', 'show_map', 'ticket_types', \
                        'user_created', 'user_type', 'venue_address', 'venue_country', 'venue_latitude', 'venue_longitude', 'venue_name', 'venue_state']
        self.drop_some_cols(todrop)
        #import ipdb; ipdb.set_trace()

        #self.drop_na()

        #self.assign_text_cluster()


        if regression:
            self.drop_columns_for_regression()
            for col in ['body_length', 'name_length', 'num_previous_payouts', 'total_quantity', 'num_ticket_types']:
                self.df[col] = scale(self.df[col])


        if not predict:
            y = self.df.pop('fraud').values
            X = self.df.values

            X_oversampled, y_oversampled = self.oversample(X, y, tp=0.3)
            return X_oversampled, y_oversampled
        if not test:
            return self.df
        # test mode, don't oversample
        self.drop_some_cols(['acct_type'])
        y = self.df.pop('fraud').values
        X = self.df.values

        return X, y
