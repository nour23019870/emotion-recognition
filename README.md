# Emotion Recognition System

A real-time facial emotion recognition system powered by CNN, OpenCV, and GPU acceleration. The application detects faces in video streams and classifies emotions into seven categories: angry, disgust, fear, happy, neutral, sad, and surprise.

![Emotion Detection Demo](img/img.jpg)

## Features

- **Real-time Emotion Classification**: Detects and classifies facial expressions in real-time
- **GPU Acceleration**: Optimized for performance with GPU support for faster inference
- **Interactive GUI**: Built with Tkinter for easy visualization and control
- **Probability Dashboard**: Real-time bar chart visualization of emotion probabilities
- **FPS Counter**: Performance monitoring with frames-per-second display

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV 4.x
- Matplotlib
- Numpy
- Pillow
- tkinter

## Installation

1. Clone this repository
```bash
git clone https://github.com/nour23019870/emotion-recognition.git
cd emotion-recognition
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download or train the emotion model
   - Use the included pre-trained model: `emotion_cnn_gpu.h5`
   - Or train your own (see Training section below)

## Usage

### Running the Application

Run the main application with default settings (using webcam):

```bash
python main.py
```

Specify a custom video source:

```bash
python main.py --source path/to/video.mp4
```

Use a different model file:

```bash
python main.py --model path/to/custom_model.h5
```

### Training Your Own Model

The project includes a training script to create your own emotion detection model:

```bash
python train_emotion_model.py
```

This will train a CNN using the dataset in the `models/train` directory and save the result as `emotion_cnn_gpu.h5`.

## Dataset Structure

The emotion dataset is organized as follows:

```
models/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Each folder contains facial images corresponding to that emotion category.

## Model Architecture

The emotion classification model is a Convolutional Neural Network with:
- Multiple convolutional layers with batch normalization
- Max pooling layers for dimensionality reduction
- Dense layers with dropout for classification
- Trained on grayscale 48x48 facial emotion images

## License

MIT License

Copyright (c) 2025 L1ght

## Acknowledgements

- Haar Cascade classifier from OpenCV for face detection
- FER-2013 Faces Database for training data

## Author

L1ght
