import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    """Load in the data

    Returns:
        df (DataFrame): Dataframe of the messages and categories
    Args:
        messages_filepath (string): Filepath to the messages csv
        categories_filepath (string): Filepath to the categories csv
    """    
 
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(left = messages, right = categories, on = 'id')
    return df


def clean_data(df):
    # Prepare data
    categories = df.categories.str.split(';', expand = True)
    row = categories.loc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str.slice(start=-1)
        categories[column] = categories[column].astype(int)

    df.drop(columns='categories', inplace=True)
    df = pd.merge(left= df, right=categories, left_on=df.index, right_on=categories.index).drop('key_0', axis=1)

    # Remove duplicates
    df.drop_duplicates(subset='id', inplace=True)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('MessagesCategorized', engine, index=False, if_exists='replace')  


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