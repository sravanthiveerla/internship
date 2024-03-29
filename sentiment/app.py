from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import re
import sklearn
import nltk
import emoji
import pickle
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer,LancasterStemmer,SnowballStemmer,WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer,PowerTransformer,StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

def emoji_remove(x):
    
    return x.apply(lambda x : emoji.demojize(x))

def decontration(x):
    
    x = x.apply(lambda x:re.sub(r"aren't", 'are not', x))
    x = x.apply(lambda x:re.sub(r"won't", 'will not', x))
    x = x.apply(lambda x:re.sub(r"doesn't", 'does not', x))
    x = x.apply(lambda x:re.sub(r"n\'t", " not", x))
    x = x.apply(lambda x:re.sub(r"\'s", " is", x))
    x = x.apply(lambda x:re.sub(r"\'d", " would", x))
    x = x.apply(lambda x:re.sub(r"\'ll", " will", x))
    x = x.apply(lambda x:re.sub(r"\'t", " not", x))
    x = x.apply(lambda x:re.sub(r"\'ve", " have", x))
    x = x.apply(lambda x:re.sub(r"\'m", " am", x))

    return x

def lowercase(x):
    
    return x.str.lower()

def html_tags(x):
    
    return x.apply(lambda x:re.sub("<.+?>"," ",x))

def urls(x):
    
    return x.apply(lambda x:re.sub("https[s]?://.+? +"," ",x))

def unwanted_characters(x):
    
    return x.apply(lambda x:re.sub("[^a-z\s]"," ",x))

def lemmatization(x):
    
    list_stp = stopwords.words("english")
    wl = WordNetLemmatizer()

    def lemmatize_text(text):
        
        words = word_tokenize(text)
        lemmatized_words = [wl.lemmatize(word, pos="v") for word in words if word not in list_stp]

        return " ".join(lemmatized_words)

    return x.apply(lemmatize_text)

preprocesser_pipe = Pipeline([("Emoji's", FunctionTransformer(emoji_remove)),
                              ('Decontration',FunctionTransformer(decontration)),
                              ('Lowercase', FunctionTransformer(lowercase)),
                              ('Html_Tags', FunctionTransformer(html_tags)),
                              ('Urls', FunctionTransformer(urls)),
                              ('Unwanted Characters', FunctionTransformer(unwanted_characters)),
                              ('Lemmatization',FunctionTransformer(lemmatization))])

bbow_pipeline = Pipeline([('Pre-Processing',preprocesser_pipe), ('Binary Bag of Words', CountVectorizer(binary= True))])


def dataframe(review):
    
    return pd.DataFrame([review], columns= ["Reviews"])


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST','GET'])
def result():

    review = request.form.get('review')
    pre = pickle.load(open(r"output_file.pkl",'rb'))
    model = pickle.load(open(r"Model.pkl",'rb') )

    if review is not None:

        query = pd.DataFrame([review], columns= ['Reviews'])
        query = pre.transform(query)
        y_pred = model.predict(query)
        
    else:
        y_pred = review
    
    return render_template('home.html', y_pred = y_pred)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')