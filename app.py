from fastapi import FastAPI

app = FastAPI(title="Ping", version="0.0.1")

@app.get("/")
def root():
    return {"ok": True}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
