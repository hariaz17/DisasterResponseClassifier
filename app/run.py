import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages_Master', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #Main breakdown of message genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #distribution of labels (1/0) for each of the 36 categories
    label_df = df.drop(columns=['message','genre','original','id'])
    #get label=1 distributions for all categories
    label_class_1 = label_df.sum()/len(label_df)
    label_class_1 = label_class_1.sort_values(ascending = False)
    #get label=0 distributions for all categories
    label_class_0 = (1-label_class_1)
    #get a list of names for the labels
    label_cats = list(label_class_1.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        
         #graph 1
            {
                'data': [
                    Bar(
                        x=label_cats,
                        y=label_class_1,
                        name = 'Label Class = 1',
                        marker = dict(
                            color = 'rgb(207, 83, 83)'
                                )
                    ),
                    Bar(
                        x=label_cats,
                        y=label_class_0,
                        name = 'Label Class = 0',
                        marker = dict(
                            color = 'rgb(167, 220, 242)'
                                )
                    )
                ],
       
                'layout': {
                          'autosize': 'true',
                            'margin': {
                                  'l': '75',
                                  'r': '50',
                                  'b': '175',
                                  't': '50',
                                  'pad': '10'
                                },
                        'title': 'Distribution of Label Classes',
                            'titlefont': {
                                'size': '20',
                                'color': 'rgb(107, 107, 107)'
                                    },
                        'yaxis': {
                        'title': "Distribution",
                            'titlefont': {
                                'size': '20',
                                'color': 'rgb(107, 107, 107)'
                                    },
                         },
                        'xaxis': {
                       
                        'title': { 
                            'text':"Categories",
                            'font': {
                                'size': '20',
                                'color': 'rgb(107, 107, 107)',
                                },
                            }
                        },
                      'barmode':'stack'
                }
                
                           
            },
        
        
        
        #graph 2
            {
                'data': [
                    Bar(
                        x=genre_names,
                        y=genre_counts,
                         marker = dict(
                            color = ['rgb(235, 152, 28)','rgb(242, 127, 114)','rgb(23, 163, 84)']
                                )
                        
                    )
                ],
       
                'layout': {
                    'title': 'Distribution of Message Genres',
                            'titlefont': {
                                'size': '20',
                                'color': 'rgb(107, 107, 107)'
                                    },
                    'yaxis': {
                        'title': "Count",
                            'titlefont': {
                                'size': '20',
                                'color': 'rgb(107, 107, 107)'
                                    },
                    },
                    'xaxis': {
                        'title': "Genre",
                            'titlefont': {
                                'size': '20',
                                'color': 'rgb(107, 107, 107)'
                                    },
                    }
                }
                           
            }
        
       
    
    
    
     
    ]
    
    
    
    
    
    
    
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()