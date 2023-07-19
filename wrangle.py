
#Import Libraries


# # default pandas decimal number display format
# pd.options.display.float_format = '{:20,.2f}'.format
#Here's the modified code with the indentation fixed:
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import env
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

# Acquire
from env import host, user, password

def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'

# Create function to retrieve zillow data
def get_zillow_data():
    '''
    This function reads in the Zillow data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    filename = 'zillow.csv'

    # Verify if file exists
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    # Search database if file doesn't exist
    else:
        sql = '''
        SELECT *
        FROM properties_2017 prop  
        INNER JOIN (
            SELECT parcelid, logerror, MAX(transactiondate) transactiondate 
            FROM predictions_2017 
            GROUP BY parcelid, logerror
        ) pred USING (parcelid) 
        LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
        LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
        LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
        LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
        LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
        LEFT JOIN storytype story USING (storytypeid) 
        LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
        WHERE prop.latitude IS NOT NULL 
        AND prop.longitude IS NOT NULL 
        AND transactiondate <= '2017-12-31'
        '''
    df = pd.read_sql(sql, get_connection('zillow'))
    df.to_csv(filename, index=False)
    return df


def remove_columns(df, cols_to_remove):  
    # remove columns not needed
    df = df.drop(columns=cols_to_remove)
    return df


def handle_missing_values(df, prop_required_column=.6, prop_required_row=.75):
    threshold = int(round(prop_required_column * len(df.index), 0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


def data_prep(df, cols_to_remove=[], prop_required_column=.6, prop_required_row=.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df


def wrangle_zillow():
    df = pd.read_csv('zillow.csv')
    
    # Restrict df to only properties that meet single unit use criteria
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    
    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt <= 1) | df.unitcnt.isnull()) \
            & (df.calculatedfinishedsquarefeet > 350)]

    # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)
    
    # Add column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange', 
                                   'Ventura'))    
    # drop columns not needed
    df = remove_columns(df, ['id', 'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid',
                             'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc',
                             'censustractandblock', 'propertylandusedesc', 'heatingorsystemdesc', 'unitcnt',
                             'buildingqualitytypeid'])
    
    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace=True)

    # Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df = df[df.calculatedfinishedsquarefeet < 8000]
    
    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()
    
    return df


def min_max_scaler(train, valid, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    train[num_vars] = scaler.fit_transform(train[num_vars])
    valid[num_vars] = scaler.transform(valid[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, valid, test


def outlier_function(df, cols, k):
    # function to detect and handle outlier using IQR rule
    for col in df[cols]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df


def get_mall_customers():
    '''
    This function reads in the Zillow data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    filename = 'mall_customers.csv'

    # Verify if file exists
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    # Search database if file doesn't exist
    else:
        sql = '''
            SELECT * 
            FROM customers
            '''
        df = pd.read_sql(sql, get_connection('mall_customers'))
        df.to_csv(filename, index=False)
        return df

def wrangle_mall_df():
    # acquire data
    sql = 'select * from customers'

    # acquire data from SQL server
    mall_df = get_mall_customers()

    # handle outliers
    mall_df = outlier_function(mall_df, ['age', 'spending_score', 'annual_income'], 1.5)

    # get dummy for gender column
    dummy_df = pd.get_dummies(mall_df.gender, drop_first=True)
    mall_df = pd.concat([mall_df, dummy_df], axis=1).drop(columns='gender')
    mall_df.rename(columns={'Male': 'is_male'}, inplace=True)

    # return mall_df
    return mall_df

#     # split the data in train, validate and test
#     # train, test = train_test_split(mall_df, train_size = 0.8, random_state = 123)
#     # train, validate = train_test_split(train, train_size = 0.75, random_state = 123)
    
#     # return min_max_scaler, train, validate, test
