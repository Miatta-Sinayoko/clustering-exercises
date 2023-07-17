
#Import Libraries
import warnings
warnings.filterwarnings("ignore")

# Wrangling
import pandas as pd
import numpy as np

# Exploring
import scipy.stats as stats

# Visualizing
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# default pandas decimal number display format
pd.options.display.float_format = '{:20,.2f}'.format

# Acquire
from env import host, user, password

# Create a function that retrieves the necessary connection URL.

def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'

# Create function to retrieve zillow data

def get_mallcustomer_data():
    '''
    This function reads in the Zillow data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    
    filename = 'zillow.csv'
    
    # Verify if file exist

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # Search database if file doesn't exist
    else:
        
        sql = '''
            SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, fips
            FROM properties_2017
            JOIN predictions_2017 ON properties_2017.parcelid = predictions_2017.parcelid
            WHERE properties_2017.propertylandusetypeid = 261 AND predictions_2017.transactiondate LIKE '2017%%';'''  
        
        df.to_csv(filename, index=False)

        df = pd.read_sql(sql, get_connection('zillow'))
        
        return df

