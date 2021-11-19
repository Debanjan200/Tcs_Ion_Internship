from flask import Flask,render_template,request
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer

model=load_model("sentiment.h5")
app=Flask(__name__)

stopwords=['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours','yourself','yourselves','he','him','his','himself','she',
 "she's",'her','hers','herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"]

def prediction(text):

    data_set=pd.read_csv("Tweets.csv")
    data_set=data_set[["airline_sentiment","text"]]

    corpus=[]

    for i in range(len(data_set)):
        sentiment=re.sub("[^a-zA-Z]",' ',data_set["text"][i])
        sentiment=sentiment.lower()
        sentiment=sentiment.split()
        ps=PorterStemmer()
        # all_stopwords=stopwords.words("english")
        sentiment=[ps.stem(word) for word in sentiment if not word in set(stopwords)]
        sentiment=' '.join(sentiment)
        corpus.append(sentiment)

    cv=CountVectorizer(max_features=1600)
    x=cv.fit_transform(corpus).toarray()

    #0->negative,1->Nutral,2->Positive
    test_corpus=[]
    
    sentiment=re.sub("[^a-zA-Z]",' ',text)
    sentiment=sentiment.lower()
    sentiment=sentiment.split()
    ps=PorterStemmer()
    # all_stopwords=stopwords.words("english")
    # stopwords.remove("not")
    sentiment=[ps.stem(word) for word in sentiment if not word in set(stopwords)]
    sentiment=' '.join(sentiment)
    test_corpus.append(sentiment)
    test=cv.transform(test_corpus).toarray()
    print(test.shape)
    pred=np.argmax(model.predict(test))

    if pred==0:
        return "Negative"

    elif pred==1:
        return "Neutral"

    else:
        return "Positive"


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def predict():
    if request.method=="POST":
        text=(request.form["sentiment"])

    predict=prediction(text)

    return render_template("index.html",prediction=predict)

if __name__=="__main__":
    app.run(debug=True)