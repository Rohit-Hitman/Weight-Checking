from flask import Flask, render_template, request
import pickle


import numpy as np


model = pickle.load(open('bmi_pickle.pkl', 'rb'))

app1 = Flask(__name__)


@app1.route('/')
def index():
    return render_template('index.html')


@app1.route('/predict', methods=['POST'])
def predict_placement():
    cgpa = float(request.form.get('height'))
    iq = int(request.form.get('weight'))
    profile_score = int(request.form.get('gender'))

    # prediction
    result = model.predict(np.array([cgpa, iq, profile_score]).reshape(1, 3))
    #print(result)
    if result[0] == 0:
        result = 'Extremely Weak'
    elif result[0] == 1:
        result = 'Weak'
    elif result[0] == 2:
        result = 'Normal'
    elif result[0] == 3:
        result = 'Overweight'
    elif result[0] == 4:
        result = 'Obesity'
    else :
        result = 'Extreme Obesity'

    return render_template("index.html", result=result)




if __name__ == '__main__':
    app1.run(host='0.0.0.0', port=8080)