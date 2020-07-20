import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
nltk.download('stopwords')

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    Load the filepath 
    Input:
        database_filepath -> path to SQlite db
    output:
        X - feature dataframe
        Y - label dataframe
        category_names - listing the columns
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('table', con=engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    tokenize and transform input text. Return cleaned text
    """
    rx = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    d_urls = re.findall(rx, text)
    for i in d_urls:
        text = text.replace(i, "urlplaceholder")
        
    # tokenize
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    
    # stemming
    stem = [PorterStemmer().stem(tok) for tok in tokens]
    
    # lemmatizing
    lem = [WordNetLemmatizer().lemmatize(tok) for tok in stem if tok not in stop_words]
    
    return lem


def build_model():
    """
    Return Grid Search model with pipeline and Classifier
    
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    
    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}
    
    cv = GridSearchCV(pipeline, parameters)
    
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    inputs
        model
        X_test
        y_test
        category_names
        
    output:
        scores
    """
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Load the data, run the model and save model
    
    """
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