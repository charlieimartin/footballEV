from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    yardline_100 = float(request.form['yardline_100'])
    down = int(request.form['down'])
    ydstogo = int(request.form['ydstogo'])

    # Predict the expected points
    input_features = np.array([[yardline_100, down, ydstogo]])
    expected_points = model.predict(input_features)[0]

    return render_template('result.html', expected_points=round(expected_points, 2))

if __name__ == "__main__":
    app.run(debug=True)