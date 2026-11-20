"""Serve cost estimates over HTTP: GET /health and POST /predict (contract and test page at /docs).

Inputs: JSON shipments with the columns of new_shipments.csv (no cost_usd), the released model at
paths.model and the validation rules in config.yaml.
Outputs: one estimated cost in USD per shipment, next to its shipment_id.
Used by: Pricing (live quotes), the smoke test and Render (uvicorn src.api:app).
Why: no ML logic here; it calls the same clean_dataframe, validate_dataframe and run_inference as
the batch pipeline, so a live estimate always equals the batch estimate for the same shipment.
"""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.clean_data import clean_dataframe
from src.config import SETTINGS
from src.infer import run_inference
from src.logger import setup_logging
from src.utils import load_model
from src.validate import validate_dataframe

logger = logging.getLogger(__name__)

SAMPLE_SHIPMENT = {
    "shipment_id": 1, "transport_mode": "Sea", "cargo_type": "General", "distance_km": 8928.0,
    "weight_kg": 1683.2, "volume_m3": 4.719, "urgent": 0, "month": 3,
}


class Shipment(BaseModel):
    """One shipment to price: the ID and feature columns of new_shipments.csv, without cost_usd.

    strict: numbers must be JSON numbers (not text) and urgent/month whole numbers;
    extra="forbid": an unknown field is an error, not silently ignored.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    shipment_id: int
    transport_mode: str
    cargo_type: str
    distance_km: float
    weight_kg: float
    volume_m3: float
    urgent: int
    month: int


class PredictionRequest(BaseModel):
    """Body of POST /predict: one or more shipments."""

    model_config = ConfigDict(
        extra="forbid", strict=True, json_schema_extra={"examples": [{"shipments": [SAMPLE_SHIPMENT]}]}
    )

    shipments: list[Shipment] = Field(min_length=1)


class Prediction(BaseModel):
    """Estimated cost of one shipment."""

    shipment_id: int
    prediction: float


class PredictionResponse(BaseModel):
    """Answer of POST /predict, in the same order as the request."""

    predictions: list[Prediction]


@asynccontextmanager
async def lifespan(api: FastAPI):
    """Load the released model once at start-up; a missing model stops the service with a clear error."""
    setup_logging(SETTINGS["paths"]["log_file"], **SETTINGS["logging"])
    api.state.model = load_model(SETTINGS["paths"]["model"])
    logger.info("API ready with model %s", SETTINGS["paths"]["model"])
    yield


app = FastAPI(
    title="Shipping cost API",
    description="Estimated carrier cost (cost_usd) per shipment for Racing Cargo Pricing. See README.md.",
    lifespan=lifespan,
    redoc_url=None,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the service is up and which model file it serves.

    Inputs: none. Outputs: {"status": "ok", "model": file name}.
    Why: Render and the smoke test check this before sending real requests.
    """
    return {"status": "ok", "model": SETTINGS["paths"]["model"].name}


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    """Estimate the cost of each shipment with the released model.

    Inputs: PredictionRequest (types and fields already checked by Pydantic).
    Outputs: PredictionResponse with shipment_id and prediction (USD) per shipment.
    Why: shipments that break a config.yaml rule (unknown category, value out of range) get 422
    with the same message as the batch pipeline, and never reach the model.
    """
    problem, features = SETTINGS["problem"], SETTINGS["features"]
    feature_columns = features["categorical"] + features["numeric"]
    df = clean_dataframe(pd.DataFrame([shipment.model_dump() for shipment in request.shipments]))
    try:
        validate_dataframe(df, [problem["id_column"]] + feature_columns, **SETTINGS["validation"])
    except ValueError as error:
        logger.warning("Request rejected: %s", error)
        raise HTTPException(status_code=422, detail=str(error)) from error
    predictions = run_inference(app.state.model, df[feature_columns])
    logger.info("Predicted %d shipments", len(predictions))
    return PredictionResponse(predictions=[
        Prediction(shipment_id=shipment_id, prediction=prediction)
        for shipment_id, prediction in zip(df[problem["id_column"]], predictions["prediction"])
    ])
