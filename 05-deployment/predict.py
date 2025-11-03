import pickle
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn


class Customer(BaseModel):
    lead_source: str
    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)



class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool


app = FastAPI(title="customer-churn-prediction")

with open("pipeline_v1.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


def predict_single(customer):
    X = dv.transform([customer])
    result = model.predict_proba(X)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())
    return PredictResponse(
        churn_probability=round(prob, 3),
        churn=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
