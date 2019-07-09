from flask import Flask, render_template, url_for, request
from sklearn.externals import joblib
from flask_restful import Resource, Api

nm_spam_model = open('nm_spam_model.pkl', 'rb')
clf_nm = joblib.load(nm_spam_model)

lr_spam_model = open('lr_spam_model.pkl', 'rb')
clf_lr = joblib.load(lr_spam_model)

app = Flask(__name__)
api = Api(app)


class Predict(Resource):
    def post(self):
        message = request.form['message']
        data = [message]
        nm_prediction = clf_nm.predict(data)
        lr_prediction = clf_lr.predict(data)

        ret = {"nm_prediction": int(nm_prediction[0]), "lr_prediction": int(lr_prediction[0])}
        return ret


api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
