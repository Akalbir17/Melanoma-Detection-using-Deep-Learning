# Melanoma Detection using Deep Learning

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)
- [Conclusion](#conclusion)

## Problem Statement
Melanoma is a type of skin cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. The goal of this project is to build an automated classification system using Convolutional Neural Networks (CNNs) to accurately detect melanoma from skin lesion images. The solution has the potential to reduce manual effort in diagnosis and assist dermatologists in the early detection of melanoma.

## Dataset
The project utilizes the Skin Cancer ISIC (International Skin Imaging Collaboration) dataset, which consists of approximately 2,500 images of skin lesions. The dataset is imbalanced, with different classes having varying numbers of images:
- Pigmented Benign Keratosis: 462 images
- Seborrheic Keratosis: 77 images
- Other classes: Variable number of images

## Methodology

### Data Preprocessing
- The dataset is loaded and preprocessed using the Augmentor library to handle class imbalance.
- Data augmentation techniques such as rotation, flipping, and zooming are applied to increase the diversity of the training data and reduce overfitting.

### Model Architecture
- A custom CNN model is built using TensorFlow and Keras.
- The model consists of three convolutional layers followed by max pooling and dropout layers.
- The output of the convolutional layers is flattened and passed through dense layers for classification.

### Training and Evaluation
- The model is trained on the augmented dataset using the Adam optimizer and categorical cross-entropy loss.
- The training data is split into training and validation sets to monitor the model's performance.
- The model is evaluated based on accuracy and loss metrics.

## Results
- The initial model trained on the imbalanced dataset suffered from overfitting, as evident from the increasing validation loss.
- After applying data augmentation and balancing the classes, the overfitting issue was significantly reduced.
- The final model achieved a training accuracy of 95.16% and a validation accuracy of 87.08%.

## Requirements
- Python 3.7+
- NumPy 1.23.1
- Pandas 1.4.3
- Matplotlib 3.5.3
- Seaborn 0.11.2
- PIL 1.1.7
- TensorFlow 2.9.0
- Augmentor 0.2.10

## Usage
1. Clone the repository: git clone https://github.com/Akalbir17/Melanoma-Detection-using-Deep-Learning.git
   
2. Install the required dependencies: `pip install -r requirements.txt`

3. Open the Jupyter Notebook `notebooks/Melanoma Detection Using CNN.ipynb`[1] and run the cells to train and evaluate the model.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Conclusion
In this project, we developed a custom CNN model for the multiclass classification of skin cancer, specifically focusing on melanoma detection. The model was trained on the Skin Cancer ISIC dataset, which presented challenges such as class imbalance and overfitting. By applying data augmentation techniques and balancing the classes, we were able to mitigate these issues and achieve a final training accuracy of 95.16% and a validation accuracy of 87.08%. The developed solution has the potential to assist dermatologists in the early detection of melanoma, reducing manual effort and improving patient outcomes.
