from flask import Flask, render_template, request 
from joblib import load
import numpy as np

app = Flask(__name__)

model = load('regression_model.joblib')


@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    square_footage = float(request.form['square_footage'])
    num_bedrooms = int(request.form['num_bedrooms'])
    num_bathrooms = int(request.form['num_bathrooms'])
    year_built = int(request.form['year_built'])
    lot_size = float(request.form['lot_size'])
    garage_size = int(request.form['garage_size'])

    user_input = np.array([[square_footage, num_bedrooms, num_bathrooms, year_built, lot_size, garage_size]])
    

    prediction = model.predict(user_input)[0]

    result = f"${prediction:,.2f}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
