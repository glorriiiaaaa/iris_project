import pickle
import pandas as pd
import numpy as np
import streamlit as st

def predict_species(sep_len, sep_width, pet_len, pet_wid, scaler_path, model_path):
    try:
        # load the scaler
        with open(scaler_path, 'rb') as file1:
            scaler = pickle.load(file1)

        # load the model
        with open(model_path, 'rb') as file2:
            model = pickle.load(file2)

        # prepare input data
        dct = {
            'SepalLengthCm': [sep_len],
            'SepalWidthCm': [sep_width],
            'PetalLengthCm': [pet_len],
            'PetalWidthCm': [pet_wid]
        }

        x_new = pd.DataFrame(dct)

        # transform input data
        xnew_pre = scaler.transform(x_new)

        # make predictions
        pred = model.predict(xnew_pre)
        probs = model.predict_proba(xnew_pre)
        max_prob = np.max(probs)

        return pred, max_prob

    except Exception as e:
        # log and display errors
        st.error(f"Error during prediction: {str(e)}")
        return None, None


# Streamlit UI
st.title("Iris Species Predictor")

# input fields for the features
sep_len = st.number_input(
    "SepalLengthCm",
    min_value=0.0,
    step=0.1,
    value=5.1
)

sep_width = st.number_input(
    "SepalWidthCm",
    min_value=0.0,
    step=0.1,
    value=3.5
)

pet_len = st.number_input(
    "PetalLengthCm",
    min_value=0.0,
    step=0.1,
    value=1.4
)

pet_wid = st.number_input(
    "PetalWidthCm",
    min_value=0.0,
    step=0.1,
    value=3.5
)

# prediction button
if st.button("Predict"):

    # file paths
    scaler_path = "Notebook/scaler.pkl"
    model_path = "Notebook/model.pkl"

    # call the prediction function
    pred, max_prob = predict_species(
        sep_len,
        sep_width,
        pet_len,
        pet_wid,
        scaler_path,
        model_path
    )

    # display results
    if pred is not None and max_prob is not None:
        st.subheader(f"Predicted Species: {pred[0]}")
        st.subheader(f"Prediction Probability: {max_prob:.4f}")
        st.progress(float(max_prob))

    else:
        st.error("Prediction Failed. Check the input values and model files.")