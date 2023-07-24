#IMPORTS 
import pandas as pd 
import env
import os
from sklearn.model_selection import train_test_split

#FUNCTIONS
# Acquire function:
url=env.get_db_url('zillow')
    
query= """SELECT *
            from properties_2017 prop
        JOIN ( SELECT parcelid, MAX(transactiondate) AS max_transactiondate
                FROM predictions_2017
                GROUP BY parcelid) trans using (parcelid)
       
        JOIN predictions_2017 pred ON trans.parcelid = pred.parcelid
                          AND trans.max_transactiondate = pred.transactiondate
        
        LEFT JOIN airconditioningtype air USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype arch USING(architecturalstyletypeid)
        LEFT JOIN buildingclasstype build USING(buildingclasstypeid)
        LEFT JOIN heatingorsystemtype heat USING(heatingorsystemtypeid)
        LEFT JOIN propertylandusetype land USING(propertylandusetypeid)
        LEFT JOIN storytype story USING(storytypeid)
        LEFT JOIN typeconstructiontype type USING(typeconstructiontypeid)
    
        WHERE land.propertylandusetypeid in (261,279)
            AND transactiondate <= '2017-12-31'
            AND prop.longitude IS NOT NULL
            AND prop.latitude IS NOT NULL
       """
directory='/Users/chellyannmoreno/codeup-data-science/clustering-exercises/'
filename=('zillow.csv')
def get_data():
    if os.path.exists(directory+filename):
        df=pd.read_csv(filename)
        df = df[df['transactiondate'].str.startswith("2017", na=False)]
        return df
    else:
        df=pd.read_sql(query,url)
        df.to_csv(filename,index=False)
        # cache data locally
        df.to_csv(filename, index=False)
        df = df[df['transactiondate'].str.startswith("2017", na=False)]
        return df
    

    #drop nulls
def handle_missing_values(df, prop_required_column, prop_required_row):
    # Drop columns based on missing value percentage
    threshold_col = int(prop_required_column * len(df.index))
    df.dropna(axis=1, thresh=threshold_col, inplace=True)
    
    # Drop rows based on missing value percentage
    threshold_row = int(prop_required_row * len(df.columns))
    df.dropna(axis=0, thresh=threshold_row, inplace=True)
    
    return df

# frunction to calculate missing values for each column:
def missing_values(df):
    # Calculate the number of missing values for each attribute
    missing_counts = df.isnull().sum()
    
    # Calculate the percentage of missing values for each attribute
    total_rows = len(df)
    missing_percentages = (missing_counts / total_rows) * 100
    
    # Create a new DataFrame with the attribute name, missing count, and missing percentage
    result_df = pd.DataFrame({
        'Attribute': missing_counts.index,
        'Missing Count': missing_counts.values,
        'Missing Percentage': missing_percentages.values
    })
    
    return result_df

#function to split data:
def split_data(df):
    # Split into train_validate and test sets
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # Split into train and validate sets
    train, validate = train_test_split(train_validate, test_size=.25, random_state=123)

    return train, validate, test
