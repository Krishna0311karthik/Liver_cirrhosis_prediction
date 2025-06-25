
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and normalizer
model = pickle.load(open("rf_acc_68.pkl", "rb"))
normalizer = pickle.load(open("normalizer.pkl", "rb"))

@app.route('/')
def index():
    return render_template('forms/index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form inputs to list of floats
        features = [float(request.form[key]) for key in request.form]
        features = np.array([features])
        features = normalizer.transform(features)

        prediction = model.predict(features)[0]
        result = "Liver Cirrhosis Detected" if prediction == 1 else "No Cirrhosis Detected"

        return render_template('forms/index.html', prediction=result)

    except Exception as e:
        return render_template('forms/index.html', prediction="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
