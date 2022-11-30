import numpy as np
from flask import Flask, request, render_template
from joblib import load
import joblib
from tensorflow.keras.models import load_model 
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import backend
from gevent.pywsgi import WSGIServer
import os

tf.keras.backend.clear_session()
app = Flask(__name__,template_folder="templates")
model=tf.keras.models.load_model("amazo.h5")
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    d = request.form['Sentence']
    print(d)
    loaded=CountVectorizer(decode_error='replace',vocabulary=joblib.load("amazo.save"))
    d=d.split("delimiter")
    result=model.predict(loaded.transform(d))
    print(result)
    prediction=result>0.5
  
    if prediction[0] == False:
    	output="Positive review"
    elif prediction[0] == True:
    	output="Negative review"
    return render_template('index.html', prediction_text='{}'.format(output)) 

port = os.getenv('VCAP_APP_PORT','5000')

if __name__ == "__main__":
    app.run(debug=False)
    app.secret_key = os.urandom(12)
    app.run(debug=False,host='0.0.0.0',port = port)
    