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
    camera_window.after(10, update_frame)

# Function to capture a photo
def capture_photo():
    ret, frame = cap.read()  # Capture a single frame
    if ret:
        # Create directory if it doesn't exist
        save_dir = "photos"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "captured_photo.jpg")
        
        cv2.imwrite(save_path, frame)  # Save the captured frame to a file
        messagebox.showinfo("Success", f"Photo captured successfully!\nSaved at: {save_path}")
    else:
        messagebox.showerror("Error", "Failed to capture photo")

# Function to start the webcam feed in a new window
def start_camera():
    global cap, camera_window, camera_label
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam")
        return

    # Create a new window for the camera feed
    camera_window = tk.Toplevel(root)
    camera_window.title("Camera Feed")

    # Create a label to display the camera feed
    camera_label = tk.Label(camera_window)
    camera_label.pack()

    # Create a button to capture the photo
    capture_button = tk.Button(camera_window, text="Capture Photo", command=capture_photo)
    capture_button.pack(pady=20)

    # Start updating the webcam feed
    update_frame()

    # Release the webcam when the window is closed
    camera_window.protocol("WM_DELETE_WINDOW", on_camera_window_close)

def on_camera_window_close():
    cap.release()
    camera_window.destroy()

# Create the main window
root = tk.Tk()
root.title("Main Window")

# Create a button to open the camera window
open_camera_button = tk.Button(root, text="Open Camera", command=start_camera)
open_camera_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()

# Send to CNN after the main loop
print("Creating the model...")
trained_model = TrainedConvNext()
print("Model created! Predicting...")
image_path = 'photos/captured_photo.jpg'
predicted_class = trained_model.predict(image_path)
print(f'Predicted class: {config.CLASS_NAMES[predicted_class]}')