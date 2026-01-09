# 🎙️ Voice Gender Recognition Using Deep Learning

This project demonstrates a **deep learning–based approach** for classifying voice gender (**male / female**) using acoustic features extracted from audio signals.  
The model is implemented using **TensorFlow/Keras** and achieves **high accuracy (~97.5%)** on the test dataset.

---

## 📌 Project Overview

The objective of this project is to build, train, and evaluate a neural network capable of predicting a speaker’s gender based on voice characteristics.

The complete pipeline includes:
- Dataset loading
- Feature preprocessing
- Model design and training
- Performance evaluation
- Real-time prediction on new audio samples

---

## 📊 Dataset

- **Source:**  
  https://cainvas-static.s3.amazonaws.com/media/user_data/cainvas-admin/voice.csv
- **Samples:** 3,168 audio samples  
- **Class Distribution:** Balanced (Male / Female)
- **Features:** 20 acoustic attributes extracted from voice signals

### Acoustic Features
meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp.ent, sfm,
mode, centroid, meanfun, minfun, maxfun, meandom, mindom,
maxdom, dfrange, modindx

---

## ⚙️ Data Preprocessing

### 1️⃣ Correlation Analysis
- Features with **correlation ≥ 0.95** were removed to reduce redundancy.
- Final input features: **17**

### 2️⃣ Label Encoding
- Target column (`label`) was **one-hot encoded**:
  - `male`
  - `female`

### 3️⃣ Train–Validation–Test Split
- **80%** Training  
- **10%** Validation  
- **10%** Testing  

### 4️⃣ Feature Scaling
- Applied **MinMaxScaler** from `sklearn.preprocessing`
- All features scaled to **[0, 1]** to ensure uniform learning

---

## 🧠 Model Architecture

A **feed-forward neural network** using TensorFlow/Keras:

| Layer Type | Units | Activation |
|-----------|-------|------------|
| Input Dense | 16 | ReLU |
| Hidden Dense | 8 | ReLU |
| Output Dense | 2 | Softmax |

### Compilation Details
- **Optimizer:** Adam (learning rate = 0.01)
- **Loss Function:** Categorical Crossentropy
- **Metric:** Accuracy

---

## 🚀 Training & Evaluation

- **Epochs:** Up to 32
- **Callbacks Used:**
  - `EarlyStopping` (patience = 5)
  - `ModelCheckpoint` (best validation loss)

### 📈 Performance
- **Test Accuracy:** ~97.48%
- **Test Loss:** ~0.074

---

## 🔍 Prediction on New Audio Files

A custom function, `extract_features_and_scale`, enables predictions on unseen audio files.

### Function Workflow
1. **Audio Loading**
   - Uses `librosa`
   - Duration limited to **5 seconds**
2. **Feature Extraction**
   - Extracts the same **17 acoustic features**
   - Uses `librosa` and `scipy.stats`
3. **Feature Scaling**
   - Uses the **same MinMaxScaler** fitted on training data
   - Values are clipped to training range to avoid extrapolation
4. **Prediction**
   - Scaled features are passed to the trained model
   - Outputs gender probabilities

---

## 🧪 Example Usage

```python
import warnings
warnings.filterwarnings('ignore')

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
    print(
        f"Prediction probability for {predicted_gender}: "
        f"{model_prediction[pred_index]:.4f}"
    )
```
## Sample Output
------------------------------------------------------
- Processing audio file: '/content/justin.wav'
- Predicted gender: female
- Prediction probabilities (male, female): [0.00106885 0.9989311 ]
- Prediction probability for female: 0.9989

------------------------------------------------------
- Processing audio file: '/content/sahuba.wav'
- Predicted gender: male
- Prediction probabilities (male, female): [0.994344   0.00565596]
- Prediction probability for male: 0.9943

------------------------------------------------------
- Processing audio file: '/content/pretty_little.wav'
- Predicted gender: female
- Prediction probabilities (male, female): [0.004059   0.99594104]
- Prediction probability for female: 0.9959

------------------------------------------------------
- Processing audio file: '/content/vikram.wav'
- Predicted gender: male
- Prediction probabilities (male, female): [0.8358242 0.1641758]
- Prediction probability for male: 0.8358

