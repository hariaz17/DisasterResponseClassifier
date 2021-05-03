import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads the messages and categories datasets
    from the provided filepaths, as a consolidated Pandas DataFrame

    Parameters
    ----------
    messages_filepath : str
      The file location of the messages dataset
    categories_filepath : str
      The file location of the categories dataset

    Returns
    ----------
    Merged Pandas DataFrame
      
    """
    try:
        #read in the datasets into separate dataframes
        messages_df = pd.read_csv(messages_filepath)
        categories_df = pd.read_csv(categories_filepath)
        
        #Merge the two datasets on the id field
        combined_df = messages_df.merge(categories_df,how='inner', on='id')
        
        return combined_df

    except:
        print("Function load_data could not be execute, please see error below:\n")
        print(sys.exc_info()[0])


def clean_data(df):
    """Performs datac-leansing on the dataframe columns
    
    Parameters
    ----------
    df : Pandas DataFrame
      Combined messages and categories dataframe
    Returns
    ----------
    Cleansed Pandas DataFrame
      
    """
    try:
        #parse the categories column by splitting into individual components
        categories = df['categories'].str.split(';',expand = True)
        
        # select the first row of the categories dataframe
        row = categories.iloc[0]
        
        # create a list of all the individual categories
        category_colnames = row.apply(lambda x: x[0:-2]).tolist()
        
        # rename the columns of `categories`
        categories.columns = category_colnames
        
        for column in categories:
            # set each value to be the last character of the string
            categories[column] = categories[column].apply(lambda x: x[-1:]).astype(int)
        
        # drop the original categories column from `df`
        df.drop(columns=['categories'],inplace=True)
        
        # concatenate the original dataframe with the new `categories` dataframe
        df = pd.concat([df,categories],axis=1, join='inner')
        
        # check number of duplicates
        df.drop_duplicates(keep='first',inplace=True)
        
        return df
    
    except:
        print("The clean_data function failed to excute:\n")
        print(sys.exc_info()[0])


def save_data(df, database_filename):
    
    """Performs datacleansing on the dataframe columns
    
    Parameters
    ----------
    df : Pandas DataFrame
      Combined messages and categories dataframe
    Returns
    ----------
    database SQL: 
        creates a database and saves the df into a table
        called 'Messages_Master'
      
    """
    try:
        engine = create_engine('sqlite:///{}'.format(database_filename))
        df.to_sql('Messages_Master',engine, index=False)
        
    except:
        print("The save_data function failed to excute:\n")
        print(sys.exc_info()[0])


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()