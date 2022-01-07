from flask import Flask,render_template,request
from keras.models import load_model
import nltk
from nltk.corpus import stopwords
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import pickle

nltk.download("stopwords")
model=load_model("my_model.h5")
cv=pickle.load(open("count_vectorizer.pkl","rb"))
encoder=pickle.load(open("encoder.pkl","rb"))
app=Flask(__name__)

def prediction(text):

    #0->anger,1->Fear,2->Joy,3->love,4->sadness,5->surprise
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
    pred=encoder.inverse_transform([np.argmax(model.predict(test))])

    return pred[0]


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