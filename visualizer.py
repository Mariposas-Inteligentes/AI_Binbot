import cv2
import os
import tkinter as tk
import configuration as config
from tkinter import font
from tkinter import messagebox
from tkinter import PhotoImage
from PIL import Image, ImageTk
from trained_convnext import TrainedConvNext

def center_window(root, width, height):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate the position of the window
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)

    # Set the geometry of the window
    root.geometry(f'{width}x{height}+{x}+{y}')

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

def show_trash(result):
    new_image = PhotoImage(file='./trashcans/Metal.gif') 
    if (result == "glass"):
        new_image = PhotoImage(file='./trashcans/Glass.gif') 
    elif (result == "cardboard"):
        new_image = PhotoImage(file='./trashcans/Cardboard.gif') 
    imaLab.config(image=new_image)
    imaLab.image = new_image

    camera_window.destroy()

# Function to capture a photo
def capture_photo():
    ret, frame = cap.read()  
    if ret:
        # Create directory if it doesn't exist
        save_dir = "photos"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "captured_photo.jpg")
        
        cv2.imwrite(save_path, frame)  # Save the captured frame to a file
        # Send to CNN after the main loop
        trained_model = TrainedConvNext()
        image_path = 'photos/captured_photo.jpg'
        predicted_class = trained_model.predict(image_path)
        show_trash(config.CLASS_NAMES[predicted_class])
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
    camera_window.geometry("750x550")
    camera_window.configure(bg = '#DAF5EB')
    center_window(camera_window, 750, 580)

    # Create a label to display the camera feed
    camera_label = tk.Label(camera_window)
    camera_label.pack()

    # Create a button to capture the photo
    capture_button = tk.Button(
        camera_window, 
        text="Capture Photo", 
        command=capture_photo,
        width=20,  
        height=2,
        font = font.Font(family="Helvetica", size=12)
    )
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
root.geometry("1209x700")
root.title("Binbot")
root.resizable(False, False)
root.configure(bg = "#DAF5EB")
center_window(root, 1209, 700)

# Default image
im=tk.PhotoImage(file='./trashcans/Closed.gif')
im= im.subsample(1,1)
imaLab= tk.Label(image=im)
imaLab.place(x=0, y=0, relwidth=1.0, relheigh=1.0)

# Create a button to open the camera window
open_camera_button = tk.Button(
    root,
    text="Open Camera", 
    command=start_camera, 
    width=20, 
    height=2,
    font = font.Font(family="Helvetica", size=12)
)
open_camera_button.pack(pady=25)

root.mainloop()

