from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved files
model = joblib.load("logistic_model.joblib")
scaler = joblib.load("scaler.joblib")
columns = joblib.load("model_columns.joblib")
label_encoder = joblib.load("label_encoder.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [int(request.form[col]) for col in columns]
        df = pd.DataFrame([input_data], columns=columns)
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        # Convert the prediction to a human-readable format
        if predicted_label == "YES":
            predicted_label = "High Risk !! Immediately consult a Pulmonologist."
        else:
            predicted_label = "Low Risk"
        return render_template('index.html', prediction_text=f"Lung Cancer Prediction: {predicted_label}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)