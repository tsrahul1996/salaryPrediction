from flask import Flask, request
from joblib import load

app = Flask(__name__)
model = load('salaryModel.pkl')


# GET REQUEST
@app.route('/')
def welcome():
    return "Welcome All"


# GET REQUEST
@app.route('/predict')
def predict_survival():
    experience = request.args.get("experience")
    prediction = model.predict([[int(experience)]])
    return "the predicted value is" + str(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

