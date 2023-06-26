import pandas as pd
df=pd.read_csv('TrainDataset.csv')

import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('form.html')


df.loc[df["v1"]=="spam",'v1']=1
df.loc[df["v1"]=="ham",'v1']=0
df_x=df["v2"] 
df_y=df["v1"]

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

tfidf_vector_second= TfidfVectorizer(min_df=1,stop_words='english')
x_traincv=tfidf_vector_second.fit_transform(x_train)
a=x_traincv.toarray()
tfidf_vector_second.inverse_transform(a[0])

mnb = MultinomialNB()
y_train=y_train.astype('int')
mnb.fit(x_traincv,y_train)

@app.route('/fun',methods=['POST'])

def fun():
	if request.method=="POST":
		name=request.form['name']
		message=request.form['msg']
		tfidf = tfidf_vector_second
		vectorize_message = tfidf.transform([message])
		x=vectorize_message.toarray()
		prediction = mnb.predict(x)

		return render_template('last.html', pred=prediction)

if __name__ == "__main__":
		app.debug=True
		app.run(threaded=False)
