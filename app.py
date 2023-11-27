# import json
# import pickle
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # Load the model
# regmodel = load_model('model.h5')
# scalar = pickle.load(open('scaled_data.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     data = request.json['data']
#     print(data)
    
#     # Convert the input data to a DataFrame and transpose it
#     features = pd.DataFrame([data])
    
#     # Convert DataFrame to numpy array
#     features_array = features.values.reshape(1, -1)
    
#     # Scale the input features
#     new_data = scalar.transform(features_array)
    
#     # Make prediction
#     output = regmodel.predict(new_data)
    
#     # Inverse transform the prediction
#     output = scalar.inverse_transform(output.reshape(-1, 1))[0]
    
#     return jsonify(float(output))

# if __name__ == "__main__":
#     app.run(debug=True)


import json
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
regmodel = load_model('model.h5')

# Load the scaler
scalar = pickle.load(open('scaled_data.pkl', 'rb'))

# Define the list of features used during training
# Adjust this list according to the features used during model training
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    
    # Convert the input data to a DataFrame
    features = pd.DataFrame([data])

    # Extract the relevant columns for scaling
    new_data = features.values.reshape(1,-1)

    # Scale the input features
    # new_data = scalar.transform(features_for_scaling)
    # Transform the data
    # new_data = scalar.transform(new_data)
    # scaled_data = scalar.fit_transform(new_data)
    # Make prediction
    output = regmodel.predict(new_data)

    # Inverse transform the prediction
    # output = scalar.inverse_transform(output.reshape(-1, 1))[0]
    output = scalar.inverse_transform(output)

    return jsonify(float(output[0]))

if __name__ == "__main__":
    app.run(debug=True)