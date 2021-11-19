from flask import Flask,render_template,request
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

model=load_model("sentiment.h5")
app=Flask(__name__)

def prediction(text):

    data_set=pd.read_csv("Tweets.csv")
    data_set=data_set[["airline_sentiment","text"]]

    corpus=[]

    for i in range(len(data_set)):
        sentiment=re.sub("[^a-zA-Z]",' ',data_set["text"][i])
        sentiment=sentiment.lower()
        sentiment=sentiment.split()
        ps=PorterStemmer()
        all_stopwords=stopwords.words("english")
        all_stopwords.remove("not")
        sentiment=[ps.stem(word) for word in sentiment if not word in set(all_stopwords)]
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
    all_stopwords=stopwords.words("english")
    all_stopwords.remove("not")
    sentiment=[ps.stem(word) for word in sentiment if not word in set(all_stopwords)]
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
