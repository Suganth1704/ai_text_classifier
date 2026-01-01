import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

data = pd.read_csv("../data/data.csv")

x = data['text']
y = data['label']

model = Pipeline(
    [
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ]
)

model.fit(x,y)
joblib.dump(model, "sentiment_model.pkl")
print("Model trained and saved successfully!")

