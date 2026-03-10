import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Class labels
classes = ["lung_aca", "lung_n", "lung_scc"]

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="lung_cancer_classifier.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Lung Cancer Detection from Histopathology Images")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img = np.array(img).astype(np.float32)

    # If you trained with ResNet preprocessing
    img = (img / 127.5) - 1

    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.4f}")

    # Show probability chart
    st.subheader("Class Probabilities")
    st.bar_chart(prediction[0])
