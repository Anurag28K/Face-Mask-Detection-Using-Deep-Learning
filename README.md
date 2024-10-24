# **Face Mask Detection Using Convolutional Neural Networks (CNN)**

## **Overview**

This project involves building a Convolutional Neural Network (CNN) model to detect whether a person in an image is wearing a face mask or not. The aim is to assist in promoting safety practices, especially in the context of global health concerns like COVID-19. The project leverages a Kaggle dataset and includes data preprocessing, model training, and evaluation steps to achieve accurate mask detection.

## **Project Highlights**

  **Dataset**: The project uses a dataset containing images of people with and without face masks.
  
  **Preprocessing**: Performed image augmentation and resizing to prepare the data for CNN modeling.
  
  **Model**: Built a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images into two categories: "Mask" and "No Mask."
  
  **Evaluation**: Assessed model performance using accuracy, precision, recall, and F1-score metrics.
  
## **Dataset Information**

   **Source**: Kaggle - Face Mask Detection Dataset

   **Size**: Contains thousands of labeled images of people wearing and not wearing masks.

|  Column Name |	      Description           |
|--------------|------------------------------|
|image	       |  The image of a person       |
|label	       |  Mask status (Mask, No Mask) |

## **Key Steps**

**Dataset Import**: Downloaded the dataset from Kaggle and loaded it into the environment.

**Data Preprocessing:**

     Resized the images to a fixed size (224x224) for uniformity.
     
     Applied image augmentation techniques such as rotation, zoom, and flipping.
     
**Model Building:**
    Implemented a CNN architecture with multiple layers (Convolution, MaxPooling, Dropout) using TensorFlow/Keras.

    Used Relu as activation functions for hidden layers and Softmax for output classification.

**Model Evaluation:**

       Evaluated the model's performance using accuracy, precision, recall, and F1-score.

## **Results**
   **Training Accuracy**: Achieved 96.5% accuracy on the training data.
   **Test Accuracy**: Achieved 95.2% accuracy on the test data.
   **Precision/Recall**: The model performed well in classifying both "Mask" and "No Mask" categories.
