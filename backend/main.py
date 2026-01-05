from fastapi import FastAPI
from pydantic import BaseModel
from backend.model import predict_transaction
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Credit Card Fraud Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Transaction(BaseModel):
    category: str
    amt: float
    gender: str
    city: str
    state: str
    zip: str
    lat: float
    long: float
    city_pop: int
    unix_time: int
    merch_lat: float
    merch_long: float

@app.post("/predict")
def predict(data: Transaction):
    return predict_transaction(data.dict())
