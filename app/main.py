from fastapi import FastAPI

app = FastAPI(title="")

@app.get("/")
def index():
    return "Hello"