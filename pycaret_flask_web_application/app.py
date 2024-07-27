from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

model = load_model('myMLmodel')
cols = ['Carat Weight', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=cols)
    
    # Hata ayıklama print ifadeleri
    print("Giriş özellikleri:", int_features)
    print("Tahmin için DataFrame:", data_unseen)
    
    try:
        prediction = predict_model(model, data=data_unseen, round=0)
        print("Tahmin DataFrame'i:", prediction)
        
        if 'prediction_label' in prediction.columns:
            prediction_value = prediction['prediction_label'][0]
            print("Tahmin Değeri:", prediction_value)
        else:
            return render_template('index.html', pred="Hata: Tahmin çıktısı 'prediction_label' sütununu içermiyor.")
    except Exception as e:
        print("Tahmin sırasında hata:", e)
        return render_template('index.html', pred="Tahmin sırasında hata.")
    
    return render_template('index.html', pred= "Price of the diamond will be $ {:,.2f}".format (prediction_value))

@app.route('/results',methods=['POST'])
def results():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction['prediction_label'][0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)