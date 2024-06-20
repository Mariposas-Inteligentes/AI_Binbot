import cv2
import os
import tkinter as tk
import configuration as config
from tkinter import messagebox
from PIL import Image, ImageTk
from trained_convnext import TrainedConvNext

# Function to update the webcam feed
def update_frame():
  ret, frame = cap.read()  # Read a frame from the webcam
  if ret:
    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the frame to a PIL image
    img = Image.fromarray(frame)
    # Convert the PIL image to an ImageTk object
    imgtk = ImageTk.PhotoImage(image=img)
    # Update the label with the new image
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)
  # Schedule the update function to run again after 10 milliseconds
  root.after(10, update_frame)

# Function to capture a photo
def capture_photo():
  ret, frame = cap.read()  # Capture a single frame
  if ret:
    # Create directory if it doesn't exist
    save_dir = "photos"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "captured_photo.jpg")
    
    cv2.imwrite(save_path, frame)  # Save the captured frame to a file
    print("Success", f"Photo captured successfully!\nSaved at: {save_path}")
  else:
    print("Error", "Failed to capture photo")

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
  messagebox.showerror("Error", "Could not open webcam")
  exit()

# Create the main window
root = tk.Tk()
root.title("Live Camera Feed")

# Create a label to display the camera feed
camera_label = tk.Label(root)
camera_label.pack()

# Create a button to capture the photo
capture_button = tk.Button(root, text="Capture Photo", command=capture_photo)
capture_button.pack(pady=20)

# Start updating the webcam feed
update_frame()

# Run the Tkinter event loop
root.mainloop()

# Release the webcam
cap.release()

# Send to CNN
print("Creating the model...")
trained_model = TrainedConvNext()
print("Model created! Predicting...")
image_path = 'photos/captured_photo.jpg'
predicted_class = trained_model.predict(image_path)
print(f'Predicted class: {config.CLASS_NAMES[predicted_class]}')
print(config.CLASS_NAMES[predicted_class] == "glass")