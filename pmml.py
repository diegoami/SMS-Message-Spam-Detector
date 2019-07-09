from sklearn.externals import joblib
from sklearn2pmml import make_pmml_pipeline


NB_spam_model = open('lr_spam_model.pkl', 'rb')
clf = joblib.load(NB_spam_model)


from sklearn2pmml import sklearn2pmml
sklearn2pmml(pipeline=make_pmml_pipeline(clf), pmml='spam_model.pmml')
print("Exported model in PMML to spam_model.pmml")

