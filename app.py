import json
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import Flask,request,app,jsonify,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=load_model('model.h5','rb')
scalar=pickle.load(open('scaled_data.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    features = np.array(list(data.values())).reshape(1, -1)
    print("Features shape:", features.shape)
    # print(np.array(list(data.values())).reshape(-1,1))
    print(features)
    # new_data=scalar.transform(np.array(list(data.values())).reshape(-1,1))
    # output=regmodel.predict(new_data)
    new_data = scalar.transform(features)
    output = regmodel.predict(new_data)
    output = scalar.inverse_transform(output.reshape(1, -1))[0]
    return jsonify(float(output))



if __name__=="__main__":
    app.run(debug=True)

