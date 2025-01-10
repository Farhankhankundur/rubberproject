import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to process the image
def process_image(image, threshold_value):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to get a binary image
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours of the white spots
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the total area of the image
    total_area = image.shape[0] * image.shape[1]

    # Calculate the area of the white spots
    white_spots_area = sum(cv2.contourArea(contour) for contour in contours)

    # Calculate the percentage of the area occupied by white spots
    percentage_area = (white_spots_area / total_area) * 100

    # Draw contours on the original image for visualization
    output_image = image.copy()
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

    return gray, binary, output_image, total_area, white_spots_area, percentage_area

# Custom CSS for styling
st.markdown(
    """
    <style>
    .header {
        background-color: #f63366;
        padding: 10px;
        border-radius: 5px;
        color: white;
        text-align: center;
        font-size: 25px;
        font-weight: bold;
    }
    .section {
        margin-top: 20px;
        padding: 10px;
        border: 1px solid #f0f0f0;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="header">White Spots Detection App</div>', unsafe_allow_html=True)

# About the App Section
with st.expander("About the App"):
    st.write(
        """
        This app detects and highlights white spots in uploaded images, calculates the total area covered by them, 
        and provides the results as a percentage. It allows you to visualize processed images and download the results.
        """
    )

# Image Upload and Processing Section
st.markdown('<div class="section">Upload and Process Your Image</div>', unsafe_allow_html=True)

# Image upload
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
threshold_value = st.slider("Set Threshold Value", min_value=0, max_value=255, value=200)

if uploaded_file is not None:
    # Load the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Process the image
    gray, binary, output_image, total_area, white_spots_area, percentage_area = process_image(image, threshold_value)

    # Display results and images
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Results")
        st.write(f"**Total area:** {total_area} pixels")
        st.write(f"**White spots area:** {white_spots_area} pixels")
        st.write(f"**Percentage of area occupied by white spots:** {percentage_area:.2f}%")

    with col2:
        st.subheader("Processed Images")
        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="Original Image with Detected White Spots")

    # Additional processed images
    col3, col4 = st.columns(2)

    with col3:
        st.image(gray, caption="Grayscale Image", channels="GRAY")

    with col4:
        st.image(binary, caption="Binary Image", channels="GRAY")

    # Option to download the processed image
    st.subheader("Download Results")
    output_pil_image = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    st.download_button(
        label="Download Image with Contours",
        data=output_pil_image.tobytes(),
        file_name="output_with_contours.png",
        mime="image/png"
    )
else:
    st.info("Please upload an image to begin.")
