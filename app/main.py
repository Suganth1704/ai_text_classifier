from fastapi import FastAPI
from model.schema import TextRequest
import joblib

app = FastAPI(title="AI Sentiment Classifier API")

model = joblib.load("./model/sentiment_model.pkl")

@app.get("/")
def index():
    return "Hello, this id AI text classifier"

@app.post("/predict")
def predict_sentiment(request:TextRequest):
    prediction = model.predict([request.text])[0]
    return{
        "text":request.text,
        "sentiment": prediction
    }