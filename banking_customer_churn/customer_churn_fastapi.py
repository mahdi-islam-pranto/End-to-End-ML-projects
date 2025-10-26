from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from typing import Literal, Annotated

app = FastAPI()

# import the ml model
with open('prediction_rfc_model.pkl', 'rb') as f:
    model = pickle.load(f)


# pydantic model to validate the input data
class InputData(BaseModel):
    credit_score: int = Field(..., description="Credit Score of the customer for the bank", ge=0)
    geography: Literal['France','Spain','Germany'] = Field(..., description="Geography of the customer for the bank")
    gender: Literal['Male','Female'] = Field(..., description="Gender of the customer for the bank")
    age: int = Field(..., ge=10, le=120, description="Age of the customer for the bank")
    tenure: int = Field(..., ge=0, le=20, description="The number of years the customer has been with the bank")
    balance: float = Field(..., ge=0, description="current bank balance in the customer's account")
    num_of_product_use: int = Field(..., ge=0, le=4, description="Number of products the customer has with the bank")
    has_credit_card: Literal['Yes','No'] = Field(..., description="Whether the customer has a credit card or not")
    is_active_member: Literal['Yes','No'] = Field(..., description="Whether the customer is an active member or not")
    estimated_salary: float = Field(..., ge=0, description="Estimated salary of the customer")


# api endpoint to predict the churn probability of a customer
@app.post("/predict_customer")
async def predict(input_data: InputData):

    # convert yes/no to 1/0 (do not mutate the pydantic model)
    converted_has_credit_card = 1 if input_data.has_credit_card.lower() == 'yes' else 0
    converted_is_active_member = 1 if input_data.is_active_member.lower() == 'yes' else 0
    

    # convert the input data to a dictionary
    input_data_dict = {
        'CreditScore': input_data.credit_score,
        'Geography': input_data.geography,
        'Gender': input_data.gender,
        'Age': input_data.age,
        'Tenure': input_data.tenure,
        'Balance': input_data.balance,
        'NumOfProducts': input_data.num_of_product_use,
        'HasCrCard': converted_has_credit_card,
        'IsActiveMember': converted_is_active_member,
        'EstimatedSalary': input_data.estimated_salary
    }

    # convert the input data to a dataframe
    input_customer_df = pd.DataFrame([input_data_dict])

    # convert categorical variables to numbers using encoders pkl file
    with open('encoders.pkl', 'rb') as f:
        loaded_encoders = pickle.load(f)

    # apply loaded encoders to input_customer df
    for column, encoder in loaded_encoders.items():
        if column in input_customer_df.columns:
            input_customer_df[column] = encoder.transform(input_customer_df[column])

    # make prediction with model pkl
    customer_prediction = model.predict(input_customer_df)

    # convert prediction target variable (0, 1) to ( Potential customer, Not a potential customer)
    if customer_prediction.tolist()[0] == 1:
        prediction = 'Not a potential customer'
    else:
        prediction = 'Potential customer'

    # get the probability of the prediction
    customer_prediction_proba = model.predict_proba(input_customer_df)

    # output of the API
    return JSONResponse(
        status_code=200,
        content={"prediction_value": customer_prediction.tolist()[0],
                 "prediction": prediction,
                 "probability": customer_prediction_proba.tolist()[0][1]
                 })
    
