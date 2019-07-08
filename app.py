from flask import Flask, render_template, url_for, request
from sklearn.externals import joblib
from flask_restful import Resource, Api

NB_spam_model = open('NB_spam_model.pkl', 'rb')
clf = joblib.load(NB_spam_model)

app = Flask(__name__)
api = Api(app)


class Predict(Resource):
    def post(self):
        message = request.form['message']
        data = [message]
        my_prediction = clf.predict(data)
        ret = {"prediction": int(my_prediction[0])}
        return ret


api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
