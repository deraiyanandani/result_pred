from fastapi import FastAPI
from joblib import load
import numpy as np

# pip install fastapi uvicorn
# python -m pip install numpy
# python -m pip install joblib
# python -m uvicorn app:app --reload

app=FastAPI()

model = load("multiclass2_api.joblib")

# testing purpose
@app.get("/")
def testing():
    return {"myapi:":"hello"}

# data predict

@app.get("/predict")
def predict(study_hours: float):

    input_data=np.array([[study_hours]])
    predict = model.predict(input_data)

    return{"study_hours":study_hours,"grade":float(predict[0])}
