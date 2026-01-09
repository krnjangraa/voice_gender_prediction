Voice Gender Recognition using Deep Learning
This project demonstrates a deep learning approach to classify voice gender (male/female) using audio features. The model is built with TensorFlow/Keras and achieves high accuracy on a given dataset.

Project Overview
The goal of this project is to build and evaluate a neural network model capable of predicting whether a voice is male or female based on various acoustic properties extracted from audio samples. The process involves data loading, preprocessing, model definition, training, and evaluation.

Dataset
The dataset used for this project is loaded from a CSV file ('https://cainvas-static.s3.amazonaws.com/media/user_data/cainvas-admin/voice.csv'). It contains 3168 voice samples, evenly split between male and female, with 20 acoustic features for each sample. The features include meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp.ent, sfm, mode, centroid, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, and modindx.

Data Preprocessing
Correlation Analysis: Highly correlated features (correlation >= 0.95) were identified and removed to reduce redundancy and potential multicollinearity. This resulted in a refined set of 17 input features.
One-Hot Encoding: The 'label' column (male/female) was one-hot encoded into two separate columns: male and female.
Data Splitting: The dataset was split into training, validation, and test sets with an 80-10-10 ratio.
Feature Scaling: MinMaxScaler from sklearn.preprocessing was applied to scale all input features to a range between 0 and 1. This ensures that features with larger values do not disproportionately influence the model during training.
Model Architecture
A simple feed-forward neural network (Sequential Model) from TensorFlow/Keras was used:

Input Layer: Dense layer with 16 units and relu activation.
Hidden Layer: Dense layer with 8 units and relu activation.
Output Layer: Dense layer with 2 units (for 'male' and 'female' classes) and softmax activation, suitable for multi-class classification probabilities.
The model was compiled with the Adam optimizer (learning rate 0.01), CategoricalCrossentropy loss function, and accuracy as the evaluation metric.

Training and Evaluation
The model was trained for up to 32 epochs with EarlyStopping (patience=5) to prevent overfitting and ModelCheckpoint to save the best model weights based on validation loss. The final model achieved a test accuracy of approximately 97.48% and a test loss of 0.074.

Prediction Functionality
A custom function extract_features_and_scale is provided to process new audio files. This function performs the following steps:

Load Audio: Loads an audio file using librosa, limiting the duration to 5 seconds for processing.
Feature Extraction: Extracts 17 acoustic features (meanfreq, sd, median, Q25, Q75, IQR, skew, sp.ent, sfm, mode, meanfun, minfun, maxfun, meandom, mindom, maxdom, modindx) using librosa and scipy.stats.
Feature Scaling: Scales the extracted features using the same MinMaxScaler instance that was fitted on the training data. Features are also clipped to the training data range before scaling to prevent extrapolation.
Prediction: Uses the trained Keras model to predict the gender based on the scaled features.
Example Usage
The notebook includes a demonstration of how to use the extract_features_and_scale function with sample audio files and obtain gender predictions.

import warnings
warnings.filterwarnings('ignore') # Suppress warnings from librosa, etc.

# List of user's 'male' recordings to process
recorded_audio_files = ['/content/justin.wav', '/content/sahuba.wav', '/content/pretty_little.wav','/content/vikram.wav']

for audio_file_path in recorded_audio_files:
    # 1. Extract and scale features from the audio file
    scaled_audio_features = extract_features_and_scale(audio_file_path, min_max_scaler, input_columns)

    # 2. Use the model to predict the gender from the scaled features
    model_prediction = model.predict(scaled_audio_features)[0]

    # 3. Determine the predicted gender
    pred_index = np.argmax(model_prediction)
    predicted_gender = output_columns[pred_index]

    # 4. Print the path of the processed audio file, the predicted gender, and the associated probability
    print(f"\n------------------------------------------------------")
    print(f"Processing audio file: '{audio_file_path}'")
    print(f"Predicted gender: {predicted_gender}")
    print(f"Prediction probabilities (male, female): {model_prediction}")
    print(f"Prediction probability for {predicted_gender}: {model_prediction[pred_index]:.4f}")
Sample Output:

------------------------------------------------------
Processing audio file: '/content/justin.wav'
Predicted gender: female
Prediction probabilities (male, female): [0.00106885 0.9989311 ]
Prediction probability for female: 0.9989
------------------------------------------------------
Processing audio file: '/content/sahuba.wav'
Predicted gender: male
Prediction probabilities (male, female): [0.994344   0.00565596]
Prediction probability for male: 0.9943
------------------------------------------------------
Processing audio file: '/content/pretty_little.wav'
Predicted gender: female
Prediction probabilities (male, female): [0.004059   0.99594104]
Prediction probability for female: 0.9959
------------------------------------------------------
Processing audio file: '/content/vikram.wav'
Predicted gender: male
Prediction probabilities (male, female): [0.8358242 0.1641758]
Prediction probability for male: 0.8358
