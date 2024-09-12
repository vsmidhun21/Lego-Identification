import os
import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas
from PIL import ImageTk, Image
from ultralytics import YOLO


class LegoPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LEGO Brick Predictor")

        # Initialize YOLO model
        self.model = YOLO('./bestTrained')

        # Create labels and buttons
        self.upload_label = Label(root, text="Upload Image:")
        self.upload_label.pack()

        self.upload_button = Button(root, text="Browse", command=self.upload_image)
        self.upload_button.pack()

        self.predict_button = Button(root, text="Predict", command=self.predict_and_display)
        self.predict_button.pack()

        # Canvas for displaying images
        self.canvas_left = Canvas(root, width=400, height=400)
        self.canvas_left.pack(side=tk.LEFT)

        self.canvas_right = Canvas(root, width=400, height=400)
        self.canvas_right.pack(side=tk.RIGHT)

        # Initialize image variables
        self.input_image = None
        self.output_image = None
        self.input_image_path = None
        self.file_name = None

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        self.file_name = os.path.basename(file_path)
        if file_path:
            self.input_image_path = file_path
            self.display_image(file_path, self.canvas_left)

    def predict_and_display(self):
        if self.input_image_path:
            results = self.model.predict(source=self.input_image_path, save=True)  # Run YOLO prediction
            output_image_path = results[0].save_dir + '/' + self.file_name # Get the path to the generated output image
            #print("Output path",output_image_path)
            self.display_image(output_image_path, self.canvas_right)

    def display_image(self, image_path, canvas):
        image = Image.open(image_path)

        # Calculate the appropriate dimensions to maintain aspect ratio
        canvas_width, canvas_height = int(canvas["width"]), int(canvas["height"])
        image_ratio = image.width / image.height
        canvas_ratio = canvas_width / canvas_height

        if image_ratio > canvas_ratio:
            # Image is wider relative to its height
            new_width = canvas_width
            new_height = int(canvas_width / image_ratio)
        else:
            # Image is taller relative to its width
            new_height = canvas_height
            new_width = int(canvas_height * image_ratio)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
        image = ImageTk.PhotoImage(image)

        canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=image)
        canvas.image = image  # Keep a reference to the image to prevent garbage collection


if __name__ == "__main__":
    root = tk.Tk()
    app = LegoPredictorApp(root)
    root.mainloop()
