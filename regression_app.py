import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model
import os

os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"


base_dir = "./"

st.image(image="images/cfl-logo.png")
st.title("Short-term predictive model")
st.header("Prediction of rail intermodal operation disruptions")
st.write(
    "This is a web app to predict the arrival delay time of the trains (in minutes)."
)
st.write(
    "Please adjust the value of each feature in the sidebar, and then click on the Predict button at the bottom to see the prediction of the model."
)


# Model setup # Using Pycaret
def predict_delay(model, df):
    predictions_data = predict_model(estimator=model, data=df)
    if predictions_data["Label"][0] >= 0:
        return "late.", predictions_data["Label"][0]
    elif predictions_data["Label"][0] < 0:
        return "early.", predictions_data["Label"][0] * (-1)


model = load_model(base_dir + "model/model")  # Using Pycaret

# Input features

teu_count = st.sidebar.slider(
    label="TEU count", min_value=0, max_value=100, value=60, step=1
)

train_length = st.sidebar.slider(
    label="Train length [m]", min_value=100, max_value=720, value=550, step=10
)

train_weight = st.sidebar.slider(
    label="Train weight [t]", min_value=290, max_value=2300, value=1200, step=10
)

wagon_count = st.sidebar.slider(
    label="Number of wagons", min_value=6, max_value=35, value=15, step=1
)

total_distance_trip = st.sidebar.slider(
    label="Total distance of the trip [km]",
    min_value=80,
    max_value=1450,
    value=650,
    step=10,
)

departure_delay = st.sidebar.slider(
    label="Departure delay [min]", min_value=-250, max_value=950, value=30, step=5
)

distance_between_control_stations = st.sidebar.slider(
    label="Distance between control stations [km]",
    min_value=1,
    max_value=800,
    value=70,
    step=5,
)

# Converting input features to a dataframe

features = {
    "teu_count": teu_count,
    "train_length": train_length,
    "train_weight": train_weight,
    "wagon_count": wagon_count,
    "total_distance_trip": total_distance_trip,
    "departure_delay": departure_delay,
    "distance_between_control_stations": distance_between_control_stations,
}

features_df = pd.DataFrame([features])
features_df["weight_per_length_of_train"] = round(
    features_df["train_weight"] / features_df["train_length"], 1
)
features_df["weight_per_wagon_of_train"] = round(
    features_df["train_weight"] / features_df["wagon_count"], 1
)
features_df.drop(columns=["train_weight", "wagon_count"], axis=1, inplace=True)

## Display of results
st.table(
    features_df.rename(
        columns={
            "teu_count": "TEU count",
            "train_length": "Train length [m]",
            "total_distance_trip": "Total distance of the trip [km]",
            "departure_delay": "Departure delay [min]",
            "distance_between_control_stations": "Distance between control stations [km]",
            "weight_per_length_of_train": "Weight per length [t/m]",
            "weight_per_wagon_of_train": "Weight per wagon [t/wagon]",
        }
    )
)

if st.button("Predict"):
    prediction = predict_delay(model, features_df)  # Using pycaret
    st.header(
        " Based on feature values, the trip will arrive "
        + str(round(prediction[1], 1))
        + " minutes "
        + str(prediction[0])
    )  #   Using pycaret
