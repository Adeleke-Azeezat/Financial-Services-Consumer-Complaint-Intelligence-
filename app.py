
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# -----------------------------------
# 1. Load the trained machine-learning model
# -----------------------------------

MODEL_PATH = Path(__file__).parent / "complaint_routing_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()


# -----------------------------------
# 2. Configure the web page
# -----------------------------------

st.set_page_config(
    page_title="Consumer Complaint Routing System",
    page_icon="📋",
    layout="centered"
)

st.title("Consumer Complaint Routing System")

st.write(
    "This application predicts the product category of a "
    "financial consumer complaint to support complaint routing."
)

st.info(
    "This is a research prototype. Staff must confirm the "
    "final routing decision."
)


# -----------------------------------
# 3. Allow the user to enter a complaint
# -----------------------------------

complaint = st.text_area(
    "Enter the consumer complaint narrative:",
    height=180,
    placeholder="Type or paste the complaint here..."
)


# -----------------------------------
# 4. Set the confidence threshold
# -----------------------------------

threshold = 0.50


# -----------------------------------
# 5. Classify the complaint
# -----------------------------------

if st.button("Classify Complaint"):

    if not complaint.strip():

        st.warning("Please enter a complaint narrative.")

    else:

        # Obtain predicted probabilities
        probabilities = model.predict_proba([complaint])[0]

        # Get product category names
        classes = model.named_steps["classifier"].classes_

        # Find the category with the highest score
        predicted_index = np.argmax(probabilities)

        predicted_product = classes[predicted_index]

        confidence = probabilities[predicted_index]


        # -----------------------------------
        # 6. Display the results
        # -----------------------------------

        st.subheader("Classification Result")

        st.write("**Suggested product category:**")

        st.success(predicted_product)

        st.write("**Highest model probability:**")

        st.write(f"{confidence:.1%}")


        # -----------------------------------
        # 7. Apply the prototype review policy
        # -----------------------------------

        if confidence >= threshold:

            st.info(
                "The highest model probability is at or above "
                "50%. This complaint is eligible for a routing "
                "suggestion, subject to staff confirmation."
            )

        else:

            st.warning(
                "The highest model probability is below 50%. "
                "Send this complaint for human review."
            )

        st.caption(
            "Note: The model probability is not a guarantee "
            "that the prediction is correct."
        )
