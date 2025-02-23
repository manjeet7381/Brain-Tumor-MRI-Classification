import streamlit as st
from PIL import Image
import numpy as np
import joblib
import cv2

@st.cache_resource
def load_model():
    model = joblib.load("tumour_det.pkl")
    return model

model = load_model()

CLASS_LABELS = {
    "glioma_tumor": "Glioma tumors are typically treated with surgery, radiation therapy, and chemotherapy.",
    "meningioma_tumor": "Meningiomas are often slow-growing and may require surgery or radiation therapy.",
    "no_tumor": "No tumor detected. Regular check-ups are recommended for preventive care.",
    "pituitary_tumor": "Pituitary tumors may require medication, surgery, or radiation therapy."
}

st.title("Brain MRI Tumor Classification")
st.subheader("Upload an MRI scan to classify the tumor type.")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    #read the image in PIL
    image = Image.open(uploaded_file)
    #Convert into numpy a array
    image = np.asarray(image)
    #Convert the image to grayscale
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    def preprocess_image(img):
        img = cv2.resize(img, (100, 100))
        
        img = img.flatten()
        img = np.array([img])  
        
        return img

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_class = prediction[0]

    st.subheader("Prediction")
    st.write(f"**{predicted_class.replace('_', ' ').title()}**")

    st.subheader("Suggestions")
    st.write(CLASS_LABELS[predicted_class])
