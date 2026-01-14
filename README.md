# 🎙️ Gender Recognition from Voice Using Deep Learning

A comprehensive **machine learning and deep learning project** that classifies a speaker’s gender (**male / female**) from voice recordings using acoustic features. The system supports **offline datasets**, **real-time audio files**, **mobile deployment**, and **hardware (Arduino) integration**.

The model is built using **TensorFlow/Keras**, achieves **~97.5% accuracy**, and is optimized for real-world deployment using **TensorFlow Lite**.

---

## 📌 Project Overview

The goal of this project is to design an end-to-end pipeline that:
- Extracts meaningful acoustic features from voice signals
- Trains a neural network to learn gender-specific voice patterns
- Performs accurate predictions on unseen audio samples
- Deploys the trained model across multiple platforms

### Key Capabilities
- Binary voice gender classification (Male / Female)
- Audio feature extraction from `.wav` files
- Real-time prediction support
- Model optimization for edge and mobile devices
- Serial communication with Arduino for hardware projects

---

## ✨ Features

### Core Functionality
- **Voice Gender Classification** using a neural network
- **Audio Feature Extraction** from speech signals
- **Real-time & File-based Predictions**
- **TensorFlow Lite Conversion** for lightweight inference
- **Arduino Integration** via serial communication

---

## 📊 Dataset

- **Source:**  
  https://cainvas-static.s3.amazonaws.com/media/user_data/cainvas-admin/voice.csv
- **Total Samples:** 3,168
- **Class Distribution:** Balanced (Male / Female)
- **Original Features:** 20 acoustic features
- **Final Features Used:** 17 (after correlation filtering)

### Acoustic Features
```
meanfreq, sd, median, Q25, Q75, IQR,
skew, kurt, sp.ent, sfm, mode,
meanfun, minfun, maxfun,
meandom, mindom, maxdom, modindx
```

---

## 🎧 Audio Features Extracted

### 1️⃣ Spectral Features (11)
- Mean Frequency (meanfreq)
- Standard Deviation (sd)
- Median Frequency (median)
- First & Third Quartile (Q25, Q75)
- Interquartile Range (IQR)
- Spectral Entropy (sp.ent)
- Spectral Flatness Measure (sfm)
- Mode Frequency
- Skewness (skew)
- Kurtosis (kurt)

### 2️⃣ Fundamental Frequency Features (4)
- Mean Fundamental Frequency (meanfun)
- Minimum Fundamental Frequency (minfun)
- Maximum Fundamental Frequency (maxfun)
- Modulation Index (modindx)

### 3️⃣ Dominant Frequency Features (3)
- Mean Dominant Frequency (meandom)
- Minimum Dominant Frequency (mindom)
- Maximum Dominant Frequency (maxdom)

---

## ⚙️ Data Preprocessing

### 1️⃣ Correlation Analysis
- Features with **correlation ≥ 0.95** were removed
- Reduced redundancy and improved generalization

### 2️⃣ Label Encoding
- Target labels were **one-hot encoded**:
  - `male`
  - `female`

### 3️⃣ Train–Validation–Test Split
- **80%** Training
- **10%** Validation
- **10%** Testing

### 4️⃣ Feature Scaling
- Applied **MinMaxScaler** from `sklearn`
- Normalized features to the range **[0, 1]**
- Clipped prediction-time features to training bounds

---

## 🧠 Model Architecture

A **feed-forward neural network** implemented using TensorFlow/Keras:

| Layer | Units | Activation |
|------|-------|------------|
| Input Dense | 16 | ReLU |
| Hidden Dense | 8 | ReLU |
| Output Dense | 2 | Softmax |

### Compilation Details
- **Optimizer:** Adam (learning rate = 0.01)
- **Loss Function:** Categorical Crossentropy
- **Metric:** Accuracy

### Regularization & Callbacks
- **EarlyStopping:** Patience = 5 epochs
- **ModelCheckpoint:** Saves best model by validation loss

---

## 🚀 Training & Performance

- **Epochs:** Up to 32
- **Test Accuracy:** ~97.48%
- **Test Loss:** ~0.074

The model demonstrates strong generalization and stable performance across unseen samples.

---

## 🔍 Prediction on New Audio Files

Predictions are made using a custom utility function:

### `extract_features_and_scale()` Workflow
1. **Audio Loading**
   - Uses `librosa`
   - Audio duration capped at **5 seconds**

2. **Feature Extraction**
   - Same **17 acoustic features** as training
   - Uses `librosa` and `scipy.stats`

3. **Feature Scaling**
   - Applies the **trained MinMaxScaler**
   - Clips values to training range

4. **Model Inference**
   - Outputs probability scores for male and female

---

## 🧪 Example Usage

```python
recorded_audio_files = [
    '/content/justin.wav',
    '/content/sahuba.wav',
    '/content/pretty_little.wav',
    '/content/vikram.wav'
]

for audio_file_path in recorded_audio_files:
    scaled_audio_features = extract_features_and_scale(
        audio_file_path, min_max_scaler, input_columns
    )

    model_prediction = model.predict(scaled_audio_features)[0]
    pred_index = np.argmax(model_prediction)
    predicted_gender = output_columns[pred_index]

    print("\n------------------------------------------------------")
    print(f"Processing audio file: '{audio_file_path}'")
    print(f"Predicted gender: {predicted_gender}")
    print(f"Prediction probabilities (male, female): {model_prediction}")
    print(f"Prediction probability for {predicted_gender}: {model_prediction[pred_index]:.4f}")
```

---

## 📌 Sample Output

```
Processing audio file: 'justin.wav'
Predicted gender: female
Prediction probabilities (male, female): [0.0011, 0.9989]

Processing audio file: 'sahuba.wav'
Predicted gender: male
Prediction probabilities (male, female): [0.9943, 0.0057]
```

---

## 📦 Deployment Options

1. **Keras Model**  
   - `gender_recognition_voice.keras`

2. **TensorFlow Lite Model**  
   - `gender_recognition_model.tflite`
   - Optimized for mobile and edge devices

3. **Arduino Integration**  
   - Serial communication for hardware-based gender indication

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Librosa
- Scikit-learn
- SciPy
- TensorFlow Lite
- Arduino (Serial Communication)

---

## 📄 License

This project is open-source and available for educational and research purposes.

