from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from src.datascience.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

# Load the model once at startup
prediction_pipeline = PredictionPipeline()

# route to display the home page
@app.route('/', methods=['GET'])  
def homePage():
    return render_template("index.html")

# Define valid ranges for each input (from experimentation in input_validation.ipynb using IQR)
valid_ranges = {
    'fixed acidity': [3.95, 12.35],
    'volatile acidity': [0.02, 1.02],
    'citric acid': [0.0, 0.91],
    'residual sugar': [0.85, 3.65],
    'chlorides': [0.04, 0.12],
    'free sulfur dioxide': [1.0, 42.0],
    'total sulfur dioxide': [6.0, 122.0],
    'density': [0.99, 1.0],
    'pH': [2.92, 3.68],
    'sulphates': [0.28, 1.0],
    'alcohol': [7.1, 13.5]
}

# route to show the predictions in a web UI
@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # Read user inputs
            user_input = {}
            for feature in valid_ranges.keys():
                val = float(request.form[feature.replace(' ', '_')])
                # Validate against min/max
                min_val, max_val = valid_ranges[feature]
                if val < min_val or val > max_val:
                    return render_template(
                        'index.html',
                        error=f"Error: {feature} must be between {min_val} and {max_val}."
                    )
                user_input[feature] = [val]

            # Convert to DataFrame with proper column names
            input_df = pd.DataFrame(user_input)

            # Make prediction
            prediction = prediction_pipeline.predict(input_df)
            pred_value = round(float(prediction[0]), 2)

            return render_template('results.html', prediction=f"{pred_value}/10")

        except Exception as e:
            print('The Exception message is:', e)
            return render_template('index.html', error="Something went wrong. Check your inputs!")

    else:
        return render_template('index.html')


if __name__ == "__main__":
	
	app.run(host="0.0.0.0", port = 8080)