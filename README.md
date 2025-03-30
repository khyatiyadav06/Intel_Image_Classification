# Intel Image Classification

## Overview
This project focuses on classifying images from the Intel Image Classification dataset into different landscape categories. The dataset consists of images of buildings, forests, glaciers, mountains, seas, and streets. The goal is to train a Convolutional Neural Network (CNN) model to accurately predict the class of a given image.

## Dataset Information
The dataset is structured into three main folders:
- **Training**
  - Buildings
  - Forest
  - Glacier
  - Mountain
  - Sea
  - Street
- **Testing**
  - Buildings
  - Forest
  - Glacier
  - Mountain
  - Sea
  - Street
- **Prediction**
  - Buildings
  - Forest
  - Glacier
  - Mountain
  - Sea
  - Street
 
  ![image](https://github.com/user-attachments/assets/f8eacc38-f8a0-49b7-9078-4ac6553819c1)


Each category contains a set of images representing different landscapes, making it a multi-class classification problem.

## Technologies Used
The project leverages the following technologies and libraries:
- **Python** - Primary programming language
- **TensorFlow/Keras** - Deep learning framework for building CNN models
- **OpenCV** - Image processing library
- **NumPy** - Efficient numerical computations
- **Pandas** - Data handling and analysis
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-Learn** - Evaluation metrics and preprocessing tools

## Data Preprocessing
Before training the model, we preprocess the dataset:
- **Resizing Images:** Standardizing image sizes to ensure consistency.
- **Normalization:** Scaling pixel values between 0 and 1 for better model performance.
- **Data Augmentation:** Applying transformations such as rotation, flipping, and zooming to enhance generalization.
- **Splitting Dataset:** Dividing the dataset into training, validation, and testing sets.

## Model Architecture
The model used for classification follows a CNN-based architecture with the following layers:
- Convolutional Layers with ReLU activation
- Max-Pooling Layers for down-sampling
- Fully Connected Dense Layers
- Dropout Layers to prevent overfitting
- Softmax Activation for multi-class classification

## Training Process
1. Load the preprocessed dataset.
2. Define the CNN model architecture.
3. Compile the model using:
   ```python
   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   ```
4. Train the model using:
   ```python
   model.fit(train_data, epochs=20, validation_data=val_data)
   ```
5. Evaluate the model performance on test data.

## Model Performance
Using the training dataset, the following results were obtained:
- **Accuracy**: 99%
- **Precision**: 13%
- **Loss**: 0.8%

![image](https://github.com/user-attachments/assets/e6be70e7-8285-4a91-aa46-e31b600b10cb)

## Credits
This project was developed as part of the Intel Image Classification challenge.
**Bootcamp:** Cognition 3.0 Machine Learning Bootcamp


