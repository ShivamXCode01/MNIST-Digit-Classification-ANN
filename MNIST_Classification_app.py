import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


# ============================================================
# 2. APPLICATION TITLE
# ============================================================

st.title("🔢 Handwritten Digit Recognition")

st.markdown(
    """
    ### Artificial Neural Network (ANN)

    Upload an image of a handwritten digit from **0 to 9**.
    The trained ANN model will predict the digit and display
    the prediction confidence and class probabilities.
    """
)

st.divider()


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "MNIST_Classification_ANN"


@st.cache_resource
def load_trained_model():
    """
    Load the trained ANN model once and cache it.
    """
    return tf.keras.models.load_model(MODEL_PATH)


try:
    if not MODEL_PATH.exists():
        st.error(
            "Model file not found! "
            "Please place mnist_ann.keras in the same folder as app.py."
        )
        st.stop()

    model = load_trained_model()

    st.success("✅ Trained ANN model loaded successfully!")

except Exception as error:
    st.error(f"Error loading the model: {error}")
    st.stop()


# ============================================================
# 4. IMAGE PREPROCESSING FUNCTION
# ============================================================

def preprocess_image(image):
    """
    Preprocess an uploaded image for MNIST digit classification.

    Expected model input:
        (1, 28, 28) for a model accepting 28x28 grayscale images.

    This preprocessing is intended for models trained on
    normalized MNIST grayscale images.
    """

    # Convert image to grayscale
    image = ImageOps.grayscale(image)

    # Resize to MNIST dimensions
    image = image.resize((28, 28))

    # Convert image to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # MNIST commonly uses a black background with a white digit.
    # Invert images that appear to have a predominantly white background.
    if np.mean(image_array) > 127:
        image_array = 255.0 - image_array

    # Normalize pixel values to the range [0, 1]
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# 5. IMAGE UPLOAD
# ============================================================

st.subheader("📤 Upload a Handwritten Digit")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"],
    help="Upload a clear image containing one handwritten digit."
)


# ============================================================
# 6. PREDICTION
# ============================================================

if uploaded_file is not None:

    try:
        # Load uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        # Display original image
        st.subheader("🖼️ Original Image")

        st.image(
            image,
            caption="Uploaded handwritten digit",
            width=250
        )

        # Preprocess image
        processed_image = preprocess_image(image)

        # Display processed image
        st.subheader("⚙️ Preprocessed Image")

        st.image(
            processed_image[0],
            caption="28 × 28 grayscale image",
            width=200,
            clamp=True
        )

        st.caption(
            "The image is converted to grayscale, resized to 28 × 28, "
            "and normalized before prediction."
        )

        st.divider()

        # Prediction button
        if st.button("🔍 Predict Digit", type="primary"):

            with st.spinner("Predicting digit..."):

                # Generate predictions
                predictions = model.predict(
                    processed_image,
                    verbose=0
                )

                # Flatten output for a standard 10-class classifier
                predictions = np.asarray(predictions).reshape(-1)

                # Check output
                if predictions.size != 10:
                    st.error(
                        f"Expected 10 output classes, "
                        f"but the model returned {predictions.size}. "
                        "Check the model architecture and output layer."
                    )
                    st.stop()

                # Get predicted class
                predicted_digit = int(np.argmax(predictions))

                # Get confidence score
                confidence = float(predictions[predicted_digit])

            # Display prediction result
            st.subheader("🎯 Prediction Result")

            st.success(
                f"### Predicted Digit: {predicted_digit}"
            )

            # Display confidence
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Predicted Digit",
                    value=predicted_digit
                )

            with col2:
                st.metric(
                    label="Prediction Score",
                    value=f"{confidence * 100:.2f}%"
                )

            st.divider()

            # Display class probabilities or scores
            st.subheader("📊 Class Probabilities / Scores")

            probability_data = {
                str(digit): float(predictions[digit])
                for digit in range(10)
            }

            st.bar_chart(probability_data)

            # Display numerical scores
            st.subheader("📋 Detailed Results")

            for digit in range(10):
                score = float(predictions[digit])

                st.write(
                    f"**Digit {digit}:** {score * 100:.2f}%"
                )

                st.progress(
                    float(np.clip(score, 0.0, 1.0))
                )

            st.divider()

            st.info(
                "Note: The displayed values represent probabilities "
                "only if the model's output layer produces probabilities, "
                "such as through softmax. Otherwise, they are model scores."
            )

    except Exception as error:
        st.error(f"Error processing the image: {error}")


else:

    st.info(
        "👆 Upload an image of a handwritten digit to start prediction."
    )


# ============================================================
# 7. SIDEBAR INFORMATION
# ============================================================

with st.sidebar:

    st.title("ℹ️ About the Project")

    st.write(
        """
        **Project:** MNIST Handwritten Digit Recognition

        **Model:** Artificial Neural Network (ANN)

        **Dataset:** MNIST

        **Input:** 28 × 28 grayscale image

        **Classes:** Digits 0–9

        **Framework:** TensorFlow / Keras

        **Interface:** Streamlit
        """
    )

    st.divider()

    st.caption(
        "This application uses a previously trained ANN model "
        "to classify handwritten digits."
    )