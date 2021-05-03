import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):

    """loads in the messages data from the provided database
       and splits the data into training data and labels
    
    Parameters
    ----------
    database_filepath : str
      full path of the database file
    
    Returns
    ----------
    X 1D - array: 
        Messages data containing texts to train the model on
    Y 2D - array:
        label data for the 36 categories to be predicted for
        each of the messages
    category_names List:
        A list containing the names of the 36 label (categories)
      
    """

    #initialise the SQL connection using the db path provided
    engine = create_engine('sqlite:///{}'.format(database_filepath)).connect()

    #read in the Messages table into a pandas dataframe
    df = pd.read_sql_table('Messages_Master',engine)
    
    #drop all rows with label = 2 for the "related column". It looks like data issue
    #All other columns only have 1 or 0 labels. Additionally, the 188 rows being dropped
    #Have 0 labels in all the other 35 categories.
    
    df = df[df['related']!=2]

    #store the messages column as the X variable to process and train the model
    X = df['message']

    #store the 36 category columns into Y variable to train and test the model predictions
    Y = df.drop(columns=['message','genre','original','id'])

    #exctract all the category names into a list to export
    category_names = list(Y.columns)


    return X, Y, category_names



def tokenize(text):

    """Loads the messages and tokenizes the text
    Parameters
    ----------
    text : str
      Text to be tokenized
    Returns
    ----------
    List of processed tokens
    
    """
    
    #Tokenise each of the messages and initialise the lemmatizer
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    #initialise an empty dictionary to store the processed tokens
    clean_tokens = []
    
    #for each of the the tokens, lematize, normalise and store in a dictionary
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    """Initialises the model by defining the model pipeline
       and using gridsearch to search of defined parameters 
    Parameters
    ----------
    None

    Returns
    ----------
    Initialsed Model ready for fitting
    
    """
    
    #Define the pipeline with Vectorising, TF-ID and the classifier
    #the classifier is RandomForest, wrapped inside a multi output classifier

    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
            ])


    #define the paramters to perform GridSearch on
    parameters = {
                'vect__max_df': (0.75,1.0),
                'vect__stop_words': ('english', None),
                'clf__estimator__n_estimators': [10,20],
                'clf__estimator__min_samples_split': [2, 5]
   
                        }

    #run GridSearchCV on the full pipeline                  
    cv = GridSearchCV(pipeline,parameters,n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    """Evaluates the performance of the training model in
       predicting the categories text sample

    Parameters
    ----------
    Model: Training Model
        model built using build_model function

    X_test: Array
        Test sample messages to predict categories for

    Y_test: Array
        Test sample of categories to evaluate the models predictions

    category_names:
        Names of the 36 categories

    Returns
    ----------
    prints full classification report for each category respectively
    
    """
    
    y_preds = model.predict(X_test)

    y_preds = pd.DataFrame(y_preds, columns=category_names)
    y_test = Y_test.reset_index(drop=True)


    i = 0
    for cat in category_names:
        i += 1

        print("Results for Category{}: {} \n {} \n".format(i,cat,classification_report(y_test[cat],y_preds[cat])))

    print("\n-----------------\nEvaluation is complete")
    #add best params


def save_model(model, model_filepath):
    """ Saves the model with the best parameters as a pickle file"""
    file_name = model_filepath
    pickle.dump(model.best_estimator_,open(file_name,'wb'))


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
              'train_classifier.py ./data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()