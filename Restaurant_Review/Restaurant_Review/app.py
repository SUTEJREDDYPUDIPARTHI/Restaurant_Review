# Importing essential libraries
from flask import Flask, render_template, request
import pickle

from pyparsing import C

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = r'C:\Users\LENOVO\Downloads\Restaurant-Reviews-Sentiment-Analysis-Deployment-master\Restaurant-Reviews-Sentiment-Analysis-Deployment-master\restaurant-sentiment-mnb-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('C:/Users/LENOVO/Downloads/Restaurant-Reviews-Sentiment-Analysis-Deployment-master/Restaurant-Reviews-Sentiment-Analysis-Deployment-master/cv-transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)