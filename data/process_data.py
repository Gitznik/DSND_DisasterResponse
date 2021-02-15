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
    """Cleans the a dataset provided as a DataFrame and returns the cleaned DataFrame.
    Cleaning includes expanding the categories and cleaning them up.


    Args:
        df (DataFrame): Data, containing categories as a single column, as well as messages

    Returns:
        DataFrame: Cleaned DataFrame
    """    
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

    # Remove rows that have a 2 in related, as this is assumed to be faulty data
    implausible_related_count = (df['related'] == 2).sum()
    df = df.loc[df.related != 2]
    print(f'Dropped {implausible_related_count} faulty messages.')
    
    return df


def save_data(df, database_filename):
    """Saves the data to a Sqlite data base

    Args:
        df (DataFrame): Final data, to be saved to a data base
        database_filename (string): Path where to create the data base
    """    
    engine = create_engine('sqlite:///' + database_filename)
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