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
df = pd.read_sql_table('cleaned_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create data frame that sums all the categories flag per genre and category
    genre_groups_freq = df.melt(id_vars=['id', 'message', 'original', 'genre']).groupby(['genre', 'variable']).agg({'value':'sum'}).reset_index()
    
    # get the top 7 sum for each category in the genre column
    df_direct = genre_groups_freq[genre_groups_freq['genre'] == 'direct'].nlargest(7, 'value')
    df_news = genre_groups_freq[genre_groups_freq['genre'] == 'news'].nlargest(7, 'value')
    df_social = genre_groups_freq[genre_groups_freq['genre'] == 'social'].nlargest(7, 'value')
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=list(df_direct['variable']),
                    y=list(df_direct['value'])
                )
            ],

            'layout': {
                'title': 'Top 7 Categories in the Direct Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories in Direct Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=list(df_news['variable']),
                    y=list(df_news['value'])
                )
            ],

            'layout': {
                'title': 'Top 7 Categories in the News Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories in News Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=list(df_social['variable']),
                    y=list(df_social['value'])
                )
            ],

            'layout': {
                'title': 'Top 7 Categories in the Social Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories in Social Genre"
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