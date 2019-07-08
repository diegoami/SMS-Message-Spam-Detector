
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv("spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# Features and Labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# Extract Feature With CountVectorizer
clf = Pipeline(memory=None,
         steps=[('cv', CountVectorizer()),
                ('mnb', MultinomialNB())], verbose=False)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Naive Bayes Classifier

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

joblib.dump(clf, 'NB_spam_model.pkl')
