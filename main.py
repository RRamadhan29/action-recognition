import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import mediapipe as mp
from collections import deque, Counter
import argparse
import json
import pyttsx3  
import threading  

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ===================== Configuration =====================
class Config:
    DATA_PATH = 'Data/'
    MODEL_PATH = 'best_model.pt'
    ACTIONS = ['hello', 'see you later', 'i or me', 'yes','no','help',
               'thank you','what?','repeat','more','fine','go to',
               'learn','sign','finish','none']  # Added 'none' class for no action
    NO_SEQUENCES = 50  # Increased number of sequences
    SEQUENCE_LENGTH = 30
    INPUT_SIZE = 1662
    HIDDEN_SIZE = 512
    NUM_LAYERS = 3
    EPOCHS = 150
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    DROPOUT = 0.4
    EARLY_STOPPING_PATIENCE = 15
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence for prediction
    SMOOTHING_WINDOW = 5  # For prediction smoothing
    MIN_GESTURE_DURATION = 0.5  # Minimum duration to register a gesture (seconds)
    COOLDOWN_PERIOD = 1.0  # Time between gesture registrations
    TTS_ENABLED = True
    
config = Config()

# ===================== Utility Functions =====================
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in 
                    results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    
    face = np.array([[res.x, res.y, res.z] for res in 
                   results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    
    lh = np.array([[res.x, res.y, res.z] for res in 
                 results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    
    rh = np.array([[res.x, res.y, res.z] for res in 
                 results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    
    return np.concatenate([pose, face, lh, rh])

def draw_styled_landmarks(image, results):
    # Draw face connections
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
    
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    # Draw hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

# ===================== Data Collection with One-Button Start =====================
def collect_data():
    # Create necessary folders
    for action in config.ACTIONS:
        for sequence in range(config.NO_SEQUENCES):
            try:
                os.makedirs(os.path.join(config.DATA_PATH, action, str(sequence)))
            except FileExistsError:
                pass

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # Reduced complexity for faster processing
    ) as holistic:
        
        for action_idx, action in enumerate(config.ACTIONS):
            if action == 'none': 
                continue  # Skip collection for 'none' class
            
            # Show start screen for each action
            start_collecting = False
            while not start_collecting:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                
                # Draw UI elements
                cv2.rectangle(frame, (0, 0), (640, 480), (50, 50, 50), -1)
                
                # Draw title
                cv2.putText(frame, f'Action: {action}', (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Total Sequences: {config.NO_SEQUENCES}', (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Sequence Length: {config.SEQUENCE_LENGTH} frames', (50, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                
                # Draw start button
                button_color = (0, 200, 0)  # Green color
                cv2.rectangle(frame, (150, 200), (490, 300), button_color, -1)
                cv2.rectangle(frame, (150, 200), (490, 300), (0, 255, 0), 3)
                cv2.putText(frame, 'PRESS SPACE TO START', (170, 260), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
                
                # Draw quit button
                cv2.rectangle(frame, (150, 350), (490, 430), (0, 0, 200), -1)
                cv2.rectangle(frame, (150, 350), (490, 430), (0, 0, 255), 3)
                cv2.putText(frame, 'PRESS Q TO QUIT', (200, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Draw progress bar
                progress_width = int(400 * (action_idx / len(config.ACTIONS)))
                cv2.rectangle(frame, (120, 450), (120 + progress_width, 470), (0, 255, 0), -1)
                cv2.rectangle(frame, (120, 450), (520, 470), (255, 255, 255), 2)
                cv2.putText(frame, f'Progress: {action_idx}/{len(config.ACTIONS)} actions', (120, 440),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1)
                if key == 32:  # Space key
                    start_collecting = True
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            print(f'Collecting data for action: {action}')
            
            # Countdown before starting all sequences
            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                cv2.rectangle(frame, (0, 0), (640, 480), (50, 50, 50), -1)
                cv2.putText(frame, f'Starting in: {countdown}', (120, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(500)
            
            # Collect all sequences for this action
            for sequence in range(config.NO_SEQUENCES):
                print(f'Collecting sequence {sequence+1}/{config.NO_SEQUENCES}')
                
                for frame_num in range(config.SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # Display info
                    cv2.rectangle(image, (0, 0), (640, 100), (0, 0, 0), -1)
                    cv2.putText(image, f'Action: {action}', (15, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Sequence: {sequence+1}/{config.NO_SEQUENCES}', (15, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Frame: {frame_num+1}/{config.SEQUENCE_LENGTH}', (400, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Progress bar for sequences
                    seq_progress = int(400 * ((sequence * config.SEQUENCE_LENGTH + frame_num + 1) / 
                                            (config.NO_SEQUENCES * config.SEQUENCE_LENGTH)))
                    cv2.rectangle(image, (120, 450), (120 + seq_progress, 470), (0, 255, 0), -1)
                    cv2.rectangle(image, (120, 450), (520, 470), (255, 255, 255), 2)
                    
                    # Extract and save keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(config.DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    
                    # Display frame
                    cv2.imshow('Data Collection', image)
                    
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                
                # Add artificial 'none' sequences
                if sequence % 5 == 0:
                    save_none_sequences(sequence // 5)
                
                print(f'Sequence {sequence+1} completed')
            
            print(f'Action {action} completed\n')
        
        cap.release()
        cv2.destroyAllWindows()

def save_none_sequences(seq_id):
    """Generate and save sequences for the 'none' class"""
    for sequence in range(5):  # Save 5 none sequences per action sequence
        for frame_num in range(config.SEQUENCE_LENGTH):
            # Generate random noise as keypoints
            keypoints = np.random.normal(0, 0.1, config.INPUT_SIZE)
            npy_path = os.path.join(config.DATA_PATH, 'none', str(seq_id * 5 + sequence), str(frame_num))
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, keypoints)

# ===================== Data Loading and Preparation =====================
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    sequences = []
    labels = []
    
    label_map = {label: num for num, label in enumerate(config.ACTIONS)}
    
    for action in config.ACTIONS:
        action_dir = os.path.join(config.DATA_PATH, action)
        if not os.path.exists(action_dir):
            continue
            
        for sequence in os.listdir(action_dir):
            sequence_path = os.path.join(action_dir, sequence)
            if not os.path.isdir(sequence_path):
                continue
                
            window = []
            for frame_num in range(config.SEQUENCE_LENGTH):
                file_path = os.path.join(sequence_path, f"{frame_num}.npy")
                if not os.path.exists(file_path):
                    break
                    
                res = np.load(file_path)
                window.append(res)
                
            if len(window) == config.SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[action])
    
    X = np.array(sequences)
    y = np.array(labels)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(config.DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long).to(config.DEVICE)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(config.DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.long).to(config.DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(config.DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.long).to(config.DEVICE)
    
    # Create datasets and dataloaders
    train_dataset = GestureDataset(X_train, y_train)
    val_dataset = GestureDataset(X_val, y_val)
    test_dataset = GestureDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

# ===================== Model Architecture =====================
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(config.DEVICE)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(config.DEVICE)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size*2)
        
        # Attention mechanism
        attn_weights = self.attention(out)  # (batch_size, seq_length, 1)
        attn_applied = torch.sum(attn_weights * out, dim=1)  # (batch_size, hidden_size*2)
        
        # Fully connected layer
        out = self.fc(attn_applied)
        return out

# ===================== Training =====================
def train_model():
    # Load data
    train_loader, val_loader, test_loader = load_data()
    
    # Initialize model
    model = AttentionLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=len(config.ACTIONS),
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print("Starting training...")
    for epoch in range(config.EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f'Epoch [{epoch+1}/{config.EPOCHS}] | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'config': vars(config)
            }, config.MODEL_PATH)
            early_stop_counter = 0
            print("Saved best model")
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Evaluate on test set
    evaluate_model(model, test_loader, config.ACTIONS)
    
    print("Training completed.")

def evaluate_model(model, test_loader, actions):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=actions, yticklabels=actions)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=actions))
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    print(f"Test Accuracy: {accuracy:.4f}")


# ===================== Text-to-Speech Function =====================
class TTSManager:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Kecepatan bicara
        self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        self.thread = None
        self.last_spoken = ""
        self.last_time = 0
        self.cooldown = 2.0  # Cooldown antara pengucapan
        
    def speak(self, text):
        """Ucapkan teks jika tidak dalam cooldown"""
        current_time = time.time()
        if text != self.last_spoken or (current_time - self.last_time) > self.cooldown:
            self.last_spoken = text
            self.last_time = current_time
            
            # Hentikan thread sebelumnya jika masih berjalan
            if self.thread and self.thread.is_alive():
                self.engine.stop()
                self.thread.join(timeout=0.1)
            
            # Mulai thread baru untuk pengucapan
            self.thread = threading.Thread(target=self._speak_thread, args=(text,))
            self.thread.daemon = True
            self.thread.start()
    
    def _speak_thread(self, text):
        """Thread untuk pengucapan teks"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {str(e)}")
    
    def stop(self):
        """Hentikan pengucapan"""
        if self.thread and self.thread.is_alive():
            self.engine.stop()
            self.thread.join(timeout=0.1)

# ===================== Real-time Recognition with Transcription and TTS =====================
def load_trained_model():
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {config.MODEL_PATH}")
    
    # Load config from saved model
    checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    saved_config = checkpoint['config']
    
    # Update current config with saved parameters
    for key, value in saved_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Initialize model
    model = AttentionLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=len(config.ACTIONS),
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully.")
    return model

class GestureTranscriber:
    def __init__(self):
        self.transcript = []
        self.current_gesture = "none"
        self.previous_gesture = "none"
        self.gesture_start_time = 0
        self.last_add_time = 0
        self.gesture_history = deque(maxlen=5)
        
    def update(self, gesture, confidence):
        current_time = time.time()
        
        # Gesture changed
        if gesture != self.current_gesture:
            # Register previous gesture if it was held long enough
            if (self.current_gesture != "none" and 
                current_time - self.gesture_start_time > config.MIN_GESTURE_DURATION):
                self._add_to_transcript(self.current_gesture)
            
            # Start timing new gesture
            self.previous_gesture = self.current_gesture
            self.current_gesture = gesture
            self.gesture_start_time = current_time
        
        # Add to transcript if cooldown period has passed
        elif (gesture != "none" and 
              current_time - self.last_add_time > config.COOLDOWN_PERIOD and
              gesture != self.transcript[-1] if self.transcript else True):
            self._add_to_transcript(gesture)
    
    def _add_to_transcript(self, gesture):
        current_time = time.time()
        
        # Avoid adding the same gesture repeatedly
        if gesture != "none" and (not self.transcript or gesture != self.transcript[-1]):
            self.transcript.append(gesture)
            self.gesture_history.append(gesture)
            self.last_add_time = current_time
            return True
        return False
    
    def get_transcript(self):
        return " ".join(self.transcript)
    
    def get_recent_gestures(self):
        return " ".join(self.gesture_history)
    
    def reset_transcript(self):
        self.transcript = []
        self.gesture_history.clear()
        self.last_add_time = 0

def realtime_recognition():
    model = load_trained_model()
    transcriber = GestureTranscriber()
    tts_manager = TTSManager() if config.TTS_ENABLED else None
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Try to set to 30fps
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    sequence = deque(maxlen=config.SEQUENCE_LENGTH)
    predictions = deque(maxlen=config.SMOOTHING_WINDOW)
    last_prediction_time = time.time()
    current_gesture = "none"
    confidence = 0.0
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            
            # Make prediction when sequence is full
            if len(sequence) == config.SEQUENCE_LENGTH:
                input_data = np.array(sequence)
                input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]
                    conf, prediction = torch.max(probabilities, 0)
                    prediction_idx = prediction.item()
                    confidence = conf.item()
                    
                    # Apply confidence threshold
                    if confidence > config.CONFIDENCE_THRESHOLD:
                        predictions.append(prediction_idx)
                    
                    # Smooth predictions with majority voting
                    if predictions:
                        smoothed_pred = Counter(predictions).most_common(1)[0][0]
                        current_gesture = config.ACTIONS[smoothed_pred]
                        last_prediction_time = time.time()
            
            # Update transcript
            transcriber.update(current_gesture, confidence)
            
            # Reset to "none" if no recent prediction
            if time.time() - last_prediction_time > 2.0:
                current_gesture = "none"
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Display prediction
            cv2.putText(image, f'Gesture: {current_gesture}', (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Confidence: {confidence:.2f}', (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Show FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(image, f'FPS: {fps:.1f}', (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display transcript
            cv2.rectangle(image, (0, 410), (640, 480), (0, 0, 0), -1)
            
            # Show recent gestures
            cv2.putText(image, 'Recent:', (20, 470),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, transcriber.get_recent_gestures(), (100, 470),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            
            # # Show full transcript
            # cv2.putText(image, 'Transcript:', (20, 200),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(image, transcriber.get_transcript(), (130, 200),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Show TTS status and instructions
            tts_status = "ON" if config.TTS_ENABLED else "OFF"
            cv2.putText(image, f"TTS: {tts_status} | Press 't' to toggle", (20, 430),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(image, "Press 'r' to reset transcript | 'q' to quit", (20, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Trigger TTS for new gestures
            if tts_manager and transcriber.gesture_history:
                last_gesture = transcriber.gesture_history[-1]
                if last_gesture != "none":
                    tts_manager.speak(last_gesture)
            
            cv2.imshow('Gesture Recognition', image)
            
            # Handle key presses
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                if tts_manager:
                    tts_manager.stop()
                break
            elif key == ord('r'):
                transcriber.reset_transcript()
            elif key == ord('t'):
                config.TTS_ENABLED = not config.TTS_ENABLED
                if config.TTS_ENABLED and not tts_manager:
                    tts_manager = TTSManager()
                elif not config.TTS_ENABLED and tts_manager:
                    tts_manager.stop()
                    tts_manager = None
        
        cap.release()
        cv2.destroyAllWindows()

# ===================== Main Function =====================
def main():
    parser = argparse.ArgumentParser(description='Sign Language Recognition System')
    parser.add_argument('mode', choices=['collect', 'train', 'realtime'], 
                       help='Mode to run: collect data, train model, or realtime recognition')
    args = parser.parse_args()
    
    if args.mode == 'collect':
        collect_data()
    elif args.mode == 'train':
        train_model()
    elif args.mode == 'realtime':
        realtime_recognition()

if __name__ == "__main__":
    main()