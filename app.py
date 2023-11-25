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
features_used_during_training = ['Duration_in_hours','Days_left','Journey_day_Friday','Journey_day_Monday',
                                 'Journey_day_Saturday','Journey_day_Sunday','Journey_day_Thursday',
                                 'Journey_day_Tuesday','Journey_day_Wednesday','Airline_Air_India',
                                 'Airline_AirAsia','Airline_AkasaAir','Airline_AllianceAir',
                                 'Airline_GO FIRST','Airline_Indigo','Airline_SpiceJet','Airline_StarAir',
                                 'Airline_Vistara','Class_Business','Class_Economy','Class_First',
                                 'Class_Premium_Economy','Source_Ahmedabad','Source_Bangalore',
                                 'Source_Chennai','Source_Delhi','Source_Hyderabad','Source_Kolkata',
                                 'Source_Mumbai','Departure_12 PM - 6 PM','Departure_6 AM - 12 PM',
                                 'Departure_After 6 PM','Departure_Before 6 AM','Total_stops_1-stop',
                                 'Total_stops_2+-stop','Total_stops_non-stop','Arrival_12 PM - 6 PM',
                                 'Arrival_6 AM - 12 PM','Arrival_After 6 PM','Arrival_Before 6 AM',
                                 'Destination_Ahmedabad','Destination_Bangalore','Destination_Chennai',
                                 'Destination_Delhi','Destination_Hyderabad','Destination_Kolkata',
                                 'Destination_Mumbai'

]

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
    features = scalar.transform(features)
    new_data = features.values.reshape(1,-1)

    # Scale the input features
    # new_data = scalar.transform(features_for_scaling)
    # Transform the data
    # new_data = scalar.transform(new_data)

    # Make prediction
    output = regmodel.predict(new_data)

    # Inverse transform the prediction
    # output = scalar.inverse_transform(output.reshape(-1, 1))[0]
    output = scalar.inverse_transform(output)

    return jsonify(float(output))

if __name__ == "__main__":
    app.run(debug=True)