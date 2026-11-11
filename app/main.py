from flask import Flask

app = Flask(__name__)

@app.get("/saude")
def saude():
    return {"status": "ok"}