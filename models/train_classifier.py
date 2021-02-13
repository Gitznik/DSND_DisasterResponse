import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name= 'MessagesCategorized', con = engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)

    lemmed = [WordNetLemmatizer().lemmatize(word) 
        for word in tokens 
        if word not in stopwords.words('english')]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    return lemmed


def build_model():
    pipeline = Pipeline(
        [
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ]
    )

    # Commented out the parameters for decreasing run time
    parameters = {
        #'clf__estimator__n_estimators': [200, 1600],
        #'clf__estimator__min_samples_split': [2, 10],
        'clf__estimator__min_samples_leaf': [1, 4],
    }

    model = GridSearchCV(pipeline, param_grid= parameters, verbose= 10)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    print(f'Best parameters: {model.best_params_}')
    y_pred = model.predict(X_test)
    for i, name in enumerate(Y_test.columns):
        report = classification_report(Y_test.iloc[:, i], pd.DataFrame(y_pred).iloc[:, i])
        print(name)
        print(report)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()