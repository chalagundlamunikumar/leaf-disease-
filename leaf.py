import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === Configuration ===
MODEL_PATH = 'p1.jpeg'
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ['Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Healthy']  # Replace with your actual class names

# === Prediction Function ===
def predict_image(img_path):
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Error", "Trained model file not found.")
        return

    model = load_model(MODEL_PATH)

    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    messagebox.showinfo("Prediction", f"✅ {predicted_class} ({confidence}%)")

# === GUI ===
def browse_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        show_image(file_path)
        predict_image(file_path)

def show_image(img_path):
    img = Image.open(img_path)
    img.thumbnail((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

# === Build GUI ===
root = tk.Tk()
root.title("🌿 Plant Leaf Disease Detector")
root.geometry("400x400")
root.resizable(False, False)

label = tk.Label(root, text="Upload a leaf image to detect disease", font=("Arial", 12))
label.pack(pady=10)

btn = tk.Button(root, text="📁 Browse Image", command=browse_image, font=("Arial", 12))
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

root.mainloop()
