# 🤟 Real-Time Sign Language Recognition

Real-time system for recognizing sign language gestures using **MediaPipe**, **LSTM with Attention**, and **Text-to-Speech (TTS)**. Built with PyTorch and OpenCV.

---

## 📌 Description

This project enables **real-time classification** of hand gestures based on sign language using **LSTM-based deep learning**, **landmark detection from MediaPipe**, and **Text-to-Speech (TTS)** for audio output.

Key features include:
- 📹 Real-time webcam input with OpenCV.
- 🧠 Temporal modeling using LSTM + Attention.
- ✋ Feature extraction using MediaPipe Hands, Pose, and Face Mesh.
- 🔊 Text-to-Speech output using `pyttsx3`.
- 🧪 Live prediction with recent history display.

---

## 🛠️ Technologies Used
**Python** 3.10

**PyTorch** – Deep learning (LSTM + Attention)

**MediaPipe** – Landmark extraction (hands, pose, face)

**OpenCV** – Real-time webcam GUI

**pyttsx3** – Offline text-to-speech

**NumPy, Matplotlib, Seaborn, Scikit-learn** – Utilities & evaluation

---

## **Project Files Structure**

```plaintext
.
├── Data/                       # Folder containing gesture sequences per class
│   ├── hello/                  # Example: gesture "hello"
│   ├── iloveyou/               # Example: gesture "I love you"
│   └── thanks/                 # Example: gesture "thanks"
├── dataCollection.py           # Data collection via webcam
├── trainer.py                  # Train LSTM + Attention model
├── main.py                     # Run real-time gesture recognition
├── best_model.pt               # Trained PyTorch model (saved weights)
├── requirements.txt            # List of required Python packages
└── README.md                   # Project documentation
```
---

## 📦 Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- PyTorch
- NumPy, matplotlib, seaborn
- pyttsx3 (untuk TTS)


```bash
pip install -r requirements.txt
```

# **🚀 Setup Instructions**

### **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RRamadhan29/action-recognition
   cd action-recognition
   ```
   
2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**:
   - Download or collect sequences of sign language gestures.
   - Place them in the `Data/` directory with each action sequence in its respective folder.

## 🎬 How to Use
A. **Collect Gesture Data**
Run this command to start collecting landmark data:
```bash
python main.py collect
```

B. **Training Data Model**
```bash
python main.py train
```
### **GPU Acceleration**

If a CUDA-compatible GPU is available, the model will automatically use it for training and inference. Ensure you have installed the appropriate versions of PyTorch and CUDA to support GPU execution.

### **Evaluating the Model**

After training, the model will evaluate the performance on the test set. The confusion matrix and probabilities for each class will be printed and visualized.

C. **Running the Model**
```bash
python main.py realtime
```
---
# **Contributing**

Contributions are welcome! If you want to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push to the branch.
5. Open a pull request.

---
