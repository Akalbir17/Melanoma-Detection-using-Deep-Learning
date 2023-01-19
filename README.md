# Melanoma-Detection-using-Deep-Learning

# Melanoma Detection Assignment
> Melanoma of skin is the 17th most common cancer worldwide. t is the 13th most common cancer in men and the 15th most common cancer in women.There were more than 150,000 new cases of melanoma of skin in 2020.Melanoma skin cancer is highly curable if it gets identified at the early stages. The first step of Melanoma skin cancer diagnosis is to conduct a visual examination of the skin's affected area. Dermatologists take the dermatoscopic images of the skin lesions by the high-speed camera, which have an accuracy of 65-80% in the melanoma diagnosis without any additional technical support. With further visual examination by cancer treatment specialists and dermatoscopic images, the overall prediction rate of melanoma diagnosis raised to 75-84% accuracy. The project aims to build an automated classification system based on image processing techniques to classify skin cancer using skin lesions images.In this project I have tried to use Convolutional Neural Network to detect Melanoma. 

## Table of Contents
* [Problem Statement](#problem-statement)
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Observations](#Observations)
* [Conclusion](#Conclusion)



## Problem Statement
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

## General Information
- The CNN model is a custom model consisting of 3 convolutional layers. 
- This model was trained on the Skin cancer ISIC The International Skin Imaging Collaboration dataset   - The dataset consisted of approximately 2500 images. 
- The model was able to achieve a training accuracy of 95.15% and a valdiation accuracy of 87.08%


## Technologies Used
- NumPy       - version 1.23. 1
- pandas      - version 1.4.3
- matplotlib  - version 3.5.3
- seaborn     - version 0.11.2 
- PIL         - version 1.1.7
- Tensorflow  - version r2.9
- Augmentor   - version 0.2.10



## Observations
- Observations from Model Creation and Building
From the visualizations it can be observed that the validation loss is increasing as the epochs increase.Indicating that the model is overfitting.To solve this issue I try to augment the data.

- Observation from the Model Building and Training of Augmented Data
From the visualizations it is quite clear that the data augmentation has reduced overfitting to quite an extent.But this augmentation has also reduced the training accuracy in the process.I will now check the distribution of the dataset.

- Observations from Class Distribution Examination
  1.  All the classes have different number of images. Therefore the dataset is imbalanced.
  2. Pigmented Benign Keratosis has the maximum number of images - 462
  3.  Seborrheic Keratosis has the minimum number of images - 77


- Observation from Model Building & training on the rectified class imbalance data
It can be seen that after rectifying the data the overfiting has drastically reduced. It can also be seen that after balancing the class the validation accuracy and training accuracy have gone up. It is a great sign! Hence the issue of overfitting and low accuracy have been resolved.


## Conclusion
Hence I have been able to create a custom convolutional neural network for multiclass classification of skin cancer. Various issues arised during the training of the model such as overfitting and class imbalnace. Both these issues were resolved by using data augmentation and the augmentor library respectively. The final training accuracy is 0.9516 and a validaton accuracy of 0.87082.
