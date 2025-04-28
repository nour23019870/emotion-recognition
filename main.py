"""
Emotion Detection with GPU-accelerated CNN, OpenCV, and Tkinter GUI.

Features:
- Tkinter GUI with video feed
- Real-time emotion probability bar chart (matplotlib)
- Improved emotion prediction display
- Modular code and error handling

Developed by: L1ght (c) 2025
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
import time
import sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def load_emotion_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


def get_video_capture(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Error: Unable to open video source {source}.")
        sys.exit(1)
    return cap


def detect_and_predict(frame, face_cascade, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    label = None
    probs = np.zeros(len(LABELS))
    if faces:
        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.expand_dims(roi_normalized, axis=(0, -1))
        prediction = model.predict(roi_reshaped, verbose=0)[0]
        label = LABELS[np.argmax(prediction)]
        probs = prediction
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({probs[np.argmax(probs)]:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame, label, probs


class EmotionApp:
    def __init__(self, root, cap, model, face_cascade):
        self.root = root
        self.cap = cap
        self.model = model
        self.face_cascade = face_cascade
        self.root.title("Emotion Detection GUI")
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.video_label = tk.Label(root)
        self.video_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.fig, self.ax = plt.subplots(figsize=(4,3))
        self.bar = self.ax.bar(LABELS, [0]*len(LABELS), color='skyblue')
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Emotion Probabilities')
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, padx=10, pady=10)
        self.fps_label = tk.Label(root, text="FPS: 0.00", font=("Arial", 12))
        self.fps_label.pack(side=tk.BOTTOM, pady=5)
        # Add Quit button
        self.quit_button = tk.Button(root, text="Quit", command=self.on_close, font=("Arial", 12), bg="#e74c3c", fg="white")
        self.quit_button.pack(side=tk.BOTTOM, pady=5)
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update)
            return
        frame, label, probs = detect_and_predict(frame, self.face_cascade, self.model)
        # FPS calculation
        self.frame_count += 1
        if self.frame_count >= 10:
            curr_time = time.time()
            self.fps = self.frame_count / (curr_time - self.prev_time)
            self.prev_time = curr_time
            self.frame_count = 0
        self.fps_label.config(text=f"FPS: {self.fps:.2f}")
        # Convert frame to Tkinter image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)
        # Update bar chart
        for rect, p in zip(self.bar, probs):
            rect.set_height(p)
        self.ax.set_ylim(0, 1)
        self.canvas.draw()
        self.root.after(10, self.update)

    def on_close(self):
        self.cap.release()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Emotion Detection with Tkinter GUI.")
    parser.add_argument('--source', type=str, default='0', help='Video source (default: 0 for webcam, or path to video file)')
    parser.add_argument('--model', type=str, default='emotion_cnn_gpu.h5', help='Path to emotion model file')
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    model = load_emotion_model(args.model)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = get_video_capture(source)
    root = tk.Tk()
    app = EmotionApp(root, cap, model, face_cascade)
    root.mainloop()


if __name__ == "__main__":
    main()
