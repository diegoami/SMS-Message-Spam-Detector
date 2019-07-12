from sklearn.externals import joblib
from sklearn2pmml import make_pmml_pipeline
import os

lr_spam_model = open( os.path.join('data','lr_spam_model.pkl'), 'rb')
clf = joblib.load(lr_spam_model)


from sklearn2pmml import sklearn2pmml
sklearn2pmml(pipeline=make_pmml_pipeline(clf), pmml=os.path.join('data','spam_model.pmml'))
print("Exported model in PMML to spam_model.pmml")

