
import pandas as pd
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn2pmml.feature_extraction.text import Splitter


df = pd.read_csv("spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# Features and Labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
df.drop(['class'], axis=1, inplace=True)
X = df['message']
y = df['label']
df.to_csv( os.path.join('data', "spam_out.csv") )

print("Saved simplified spam data to spam_out.csv")

# Extract Feature With CountVectorizer
clf_lr = Pipeline(memory=None,
         steps=[('cv', CountVectorizer(tokenizer=Splitter())),
                ('lr', LogisticRegression())], verbose=False)

clf_nb = Pipeline(memory=None,
         steps=[('cv', CountVectorizer(tokenizer=Splitter())),
                ('nm', MultinomialNB())], verbose=False)

def train_and_save_model(clf, model_name, model_save_file):
    print(f"====== {model_name} =========")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)
    print("Test Accuracy")
    print(clf.score(X_test, y_test))
    y_test_pred = clf.predict(X_test)
    print("Test Precision, Recall, F1")
    print(precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), f1_score(y_test, y_test_pred))
    print("Test Confusion Matrix")
    print(confusion_matrix(y_test, y_test_pred))
    print("Overall Accuracy")
    print(clf.score(X, y))
    y_pred = clf.predict(X)
    print("Overall Precision, Recall, F1")

    print(precision_score(y, y_pred), recall_score(y, y_pred), f1_score(y, y_pred))
    print("Overall Confusion Matrix")
    print(confusion_matrix(y, y_pred))
    joblib.dump(clf, model_save_file)
    print(f"Dumped model to {model_name}")


train_and_save_model(clf_lr, "logistic_regression", os.path.join('data',"lr_spam_model.pkl"))
train_and_save_model(clf_nb, "multinomial naive gaussian", os.path.join('data',"nm_spam_model.pkl"))
