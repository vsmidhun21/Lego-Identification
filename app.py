import streamlit as st
from PIL import Image
import os
import io
import base64
import tempfile
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('./bestTrained.pt')

def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

st.title('LEGO Brick Predictor')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_file_path = temp_file.name

    # Run YOLO prediction using the saved image file
    results = model.predict(source=temp_file_path, save=True)
    output_image_path = os.path.join(results[0].save_dir, os.path.basename(temp_file_path))
    output_image = Image.open(output_image_path)

    # Display prediction results
    st.subheader('Predicted LEGO Bricks')
    st.image(output_image, caption='Predicted Bricks', use_column_width=True)

    # Extracting color and size from the results
    num_bricks = len(results[0].boxes)
    brick_details = []

    for box in results[0].boxes:
        # Convert label tensor to string and extract color and size
        label = model.names[int(box.cls)]  # Convert the class index to its string label
        color, size = label.split("_")  # Split the label into color and size
        brick_details.append({'color': color, 'size': size})

    st.write(f'Number of Bricks: {num_bricks}')
    st.write('Brick Details:')
    for brick in brick_details:
        st.write(f'Color: {brick["color"]}, Size: {brick["size"]}')

    # Provide a download link for the output image
    st.markdown(get_image_download_link(output_image, 'predicted_image.jpg', 'Download the output image'), unsafe_allow_html=True)

    # Clean up the temporary file
    os.remove(temp_file_path)
