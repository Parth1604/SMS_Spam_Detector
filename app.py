from flask import Flask,render_template,request
import pickle,joblib

model = pickle.load(open('model.pkl','rb'))
vectorizer = joblib.load('vectorizer.pkl')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict_placement():
    message = request.form.get('msg')
    vect_message = vectorizer.transform([message])
    temp=vect_message.toarray()
    result_prediction = model.predict(temp)

    if result_prediction[0] == 1:
        return render_template('spam.html')
    else:
        return render_template('ham.html') 

if __name__ == '__main__':
    app.run(debug=True)