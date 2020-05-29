import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib

def load_data(database_filepath):
    """
    load data from specified path
    Args:
        database_filepath: str
    Returns:
        tuple of the feature and labels dataframes and list of the categories unique names
    """
    # load data from the database filename
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('cleaned_data', engine)
    
    # split the dataframe to features and labels
    X = df['message']
    Y = df[df.columns[4:]]
    return X, Y, Y.columns.tolist()

def tokenize(text):
    """
    normalize the text to all lower case and remove any non letter or numbers characters then add nltk lemmatization to each text token

    Args:
        text: str
    Return:
        list of cleansed and lemmatized tokens
    """
    # lowercase the text and keep text that only contains letters and numbers
    text = re.sub('[^a-z0-9]', ' ', text.lower())

    # split text to list
    tokens = nltk.word_tokenize(text.strip())

    # applying nltk lemmatization to each token
    lemmatizer = nltk.WordNetLemmatizer() 
    tokens_lemmatized = [lemmatizer.lemmatize(tok) for tok in tokens]

    return tokens_lemmatized


def build_model():
    """
    it create sklearn pipeline to create tf-idf vectors then pass the result to multi output classifier that use random forest to 
    build a model for mutli label classification

    Returns:
        pipeline object that has both fit and predict methods. 
    """
    # create sklearn pipeline to train Random forest model on tf-idf vectors
    return Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split = 2)))
    ])


def evaluate_model(model, X_test, Y_test, category_names):
    """
    predict the model output on the test data and evaluate the result using 'classification_report' in sklearn
    Args:
        model: sklearn estimator
        X_test: pandas dataframe
        Y_test: pandas dataframe
        category_names: list:
    Returns:
        it print the f-score, precision and recall for each column in the model prediciton output.
    """

    # get prediction on the test data
    y_pred = model.predict(X_test)

    # display f-score, precision, recall for each column in the labels
    for i, column_name in enumerate(category_names):
        print(f'This Classification Report for {column_name}')
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))
        print('\n')


def save_model(model, model_filepath):
    """
    save an sklearn model to pkl file for future loading
    Args:
        model: sklearn estimator
        model_filepath: str: path to save the model
    Returns:
        it saves the input model to the specified path
    """
    # save the model in pkl format
    joblib.dump(model, model_filepath)


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