import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load files from csv format to pandas dataframe
    Args:
        messages_filepath: str: path to the messages csv file
        categories_filepath: str: path to the categories csv file
    Returns:
        pandas dataframe that combine both files using their ids
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, how='outer', on=['id', 'id'])

    return df

def clean_data(df):
    """
    clean the categories column by separating each value in it into single column then specifiying approapriate column name.
    In addition create flags for each category column of 1 or 0 and change its datatype

    Args:
        df: pandas dataframe: containg 'categories' column
    Returns:
        pandas dataframe that has each value in the categories column replaced and expanded into single column.
    """
    categories = df['categories'].str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = [val.split('-')[0] for val in list(row.values)]
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x.split('-')[1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int) 

    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    # remove duplicate rows in the dataframe
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    save pandas dataframe to a sqlite database.
    
    Args:
        df: pandas dataframe: 
        database_filename: str: sqlite database filename 
    Returns:
        it creates an sqlite database with the specified name and save the dataframe in table called 'cleaned_data'
    """
    # create database in sqlite dbms and table called 'cleaned_data' contains the df
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('cleaned_data', engine, index=False, if_exists = 'replace')  


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