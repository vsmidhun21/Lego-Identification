import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import tempfile
import os

# Load YOLO model (YOLOv8)
model = YOLO('./FinalCoShSi.pt')

# YOLO detection function
def detect_and_draw_boxes(image):
    # Run YOLO object detection on the image/frame
    results = model(image)

    # Extract bounding boxes and labels, and draw on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            label = result.names[int(box.cls[0])]  # Object label

            # Draw bounding box and label on the frame
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return image

# Define a class to process the live video frames for WebRTC
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = detect_and_draw_boxes(img)
        return img

# Streamlit App Interface
st.title("LEGO Brick Detection")

# Choose input format
input_option = st.radio("Choose input format", ("Upload Image/Video", "Live Camera", "Capture from Camera"))

if input_option == "Upload Image/Video":
    upload_option = st.radio("Upload Image or Video", ("Image", "Video"))

    if upload_option == "Image":
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Convert the image to RGB mode if it's in RGBA (PNG with alpha channel)
            if image.mode == 'RGBA':
                image = image.convert('RGB')

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
                label = model.names[int(box.cls)]
                color, shape, size = label.split("_")
                brick_details.append({'color': color, 'shape': shape, 'size': size})

            st.write(f'Number of Bricks: {num_bricks}')
            st.write('Brick Details:')
            for brick in brick_details:
                st.write(f'COLOR: {brick["color"]},  SHAPE: {brick["shape"]},  SIZE: {brick["size"]}')

            # Clean up the temporary file
            os.remove(temp_file_path)

    elif upload_option == "Video":
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)

            # Streamlit placeholder for video
            frame_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run YOLO object detection on the frame
                frame = detect_and_draw_boxes(frame)

                # Display the frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb)

            cap.release()

elif input_option == "Live Camera":
    # Use streamlit-webrtc for real-time camera stream
    webrtc_streamer(key="YOLO", video_transformer_factory=YOLOVideoTransformer)

elif input_option == "Capture from Camera":
    # Capture a photo using Streamlit's camera_input
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        img = Image.open(camera_image)

        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            img.save(temp_file.name)
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
            label = model.names[int(box.cls)]
            color, shape, size = label.split("_")
            brick_details.append({'color': color, 'shape': shape, 'size': size})

        st.write(f'Number of Bricks: {num_bricks}')
        st.write('Brick Details:')
        for brick in brick_details:
            st.write(f'COLOR: {brick["color"]},  SHAPE: {brick["shape"]},  SIZE: {brick["size"]}')

        # Clean up the temporary file
        os.remove(temp_file_path)