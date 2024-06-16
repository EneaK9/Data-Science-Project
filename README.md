![Polis Logo](/assets/logopolis.jpg)

# Data Science Project

## Water Bottle Image Classification Project

# Description

This project is centered around developing a robust image classification system capable of accurately categorizing images of water bottles into three distinct classes using Convolutional Neural Networks (CNN). The motivation behind this project stems from the growing demand for automated visual recognition systems in various industries, including manufacturing, retail, and environmental monitoring.

### Objectives

1.	**Accurate Classification**: 

      The primary objective is to achieve high accuracy in identifying and classifying different types of water bottles based on their visual attributes captured in images. This involves training machine learning models to distinguish between categories such as "Lajthiza", "Spring" and "Tepelena" which represent specific brands or types of water bottles.

2.	**Model Evaluation**: 
    - **Evaluate and compare the performance of three distinct classifiers**:
    - **Baseline Model (Dense Layers Only)**: 
      - A straightforward architecture using densely connected layers for classification.
    - **Convolutional Model**: 
      - Incorporating convolutional layers to extract spatial features from images, which is effective for capturing patterns and structures in visual data.
    - **Pre-trained Model (MobileNetV2)**: 
      - Utilizing a pre-trained MobileNetV2 model, leveraging transfer learning to capitalize on patterns learned from a large-scale dataset (ImageNet) and fine-tuning it for the specific task of water bottle classification.

3.	**Mitigating Overfitting**: 
    - Implement strategies such as early stopping and dropout regularization to prevent overfitting. Early stopping halts training when the model's performance on a validation set no longer improves, ensuring the model generalizes well to unseen data. Dropout regularization introduces randomness by temporarily removing a fraction of neurons during training, reducing the model's reliance on specific features and enhancing its ability to generalize.

4.	**Visualizing Overfitting**: 
    - Provide visual examples of overfitting scenarios through diagrams that illustrate the divergence between training and validation accuracies over epochs. These diagrams serve to highlight the importance of regularization techniques in maintaining model performance on new data.


### Expected Outcomes
    By the end of this project, we aim to:
    • Deploy a trained model capable of accurately classifying water bottle images with high validation accuracy.
    • Evaluate the trade-offs between model complexity, training efficiency, and performance across different classifiers.
    • Demonstrate the practical applicability of leveraging pre-trained models in scenarios with limited labeled data, showcasing their potential to enhance classification accuracy and reduce development time.
    
    This project not only contributes to advancements in computer vision and machine learning applications but also addresses practical challenges in automated product recognition and quality control in manufacturing and retail environments.

### Classifiers Overview
1.	**Baseline Model (Dense Layers Only)**:
    - The baseline model employs a straightforward architecture consisting solely of densely connected layers, also known as fully connected layers.
    - It takes flattened input images and passes them through several dense layers with ReLU activation functions, which help in learning non-linear mappings between input features and class labels.
    - This model serves as a fundamental benchmark, providing a baseline validation accuracy of 90%.
    - To mitigate overfitting, dropout regularization with a rate of 0.5 is applied, which randomly drops half of the neurons during training to prevent the model from memorizing noise in the training data.

2.	**Convolutional Model**:
    - The convolutional model incorporates convolutional layers, which are pivotal in capturing spatial hierarchies of features from images.
    - It begins with convolutional layers that apply filters to extract features such as edges, textures, and patterns from the input images.
    - Each convolutional layer is followed by a max-pooling layer to downsample the feature maps, reducing the computational complexity while retaining important features.
    - This architecture is well-suited for image classification tasks, as it learns hierarchical representations of the input data.
    - The convolutional model achieves a significantly improved validation accuracy of 93%, demonstrating the efficacy of convolutional neural networks in extracting meaningful features from water bottle images.

3.	**Pre-trained Model (MobileNetV2)**:
    - The pre-trained MobileNetV2 model utilizes transfer learning, leveraging knowledge from a model pre-trained on the large-scale ImageNet dataset.
    - MobileNetV2 is chosen for its lightweight architecture and high performance in mobile and embedded vision applications.
    - In this approach, the MobileNetV2 model is used as a feature extractor, where its convolutional base is frozen to prevent further training and preserve the learned features.
    - Global average pooling is applied to reduce the spatial dimensions of the feature maps produced by MobileNetV2, followed by dense layers with softmax activation for classification.
    - By fine-tuning the pre-trained model on the specific task of water bottle classification, this approach achieves the highest validation accuracy of 95%.
    - This demonstrates the effectiveness of transfer learning in adapting pre-existing knowledge to solve a new classification task efficiently.

### Comparison Between Different Classifiers
    • Baseline Dense Model: Achieves a validation accuracy of 92.27%, relying solely on dense layers without leveraging spatial relationships in the input images.
    • Convolutional Model: Achieves a validation accuracy of 99.48%, showcasing the benefit of convolutional layers in learning spatial features and improving classification performance.
    • Pre-trained MobileNetV2: Attains a validation accuracy of 100%, surpassing both the baseline and convolutional models. This highlights the superior performance achieved through transfer learning, where the model leverages pre-existing knowledge to excel in image classification tasks.


## Handling Overfitting and Underfitting

### Resolution Strategies

1.	**Early Stopping**:
    - Overfitting occurs when a model learns to memorize the training data rather than generalize to unseen data. Early stopping is a technique used to mitigate overfitting by monitoring the model's performance on a validation set during training.
    - During training, if the validation loss stops improving or begins to degrade after a certain number of epochs, early stopping halts the training process to prevent further overfitting.
    - In our project, early stopping is implemented by monitoring the validation loss. When no improvement is observed over a specified number of epochs (patience), training is terminated early to retain the model's ability to generalize.
2.	**Dropout Regularization**:
    - Dropout regularization is a technique used to reduce overfitting by randomly dropping a fraction of neurons (along with their connections) during training.
    - By randomly deactivating neurons, dropout introduces noise into the learning process, forcing the model to learn redundant representations and reducing reliance on specific features.
    - In our models, dropout layers with dropout rates of 0.5 and 0.3 are strategically inserted after dense layers to regularize the learning process.
    - This technique enhances the model's ability to generalize by making it more robust and less prone to memorizing noise in the training data.


### Examples of Diagrams Showing Overfitting
<!-- ![Overfitting Diagram](overfitting_diagram.png) -->

The diagram illustrates a typical scenario of overfitting where training accuracy continues to improve while validation accuracy plateaus or declines.

## Comparison Between Different Classifiers

| Classifier           | Validation Accuracy | Key Features                       |
|----------------------|---------------------|------------------------------------|
| Baseline Dense       | 92.27%%             | Dense layers only                  |
| Convolutional        | 99.48%%             | Convolutional layers added         |
| Pre-trained MobileNet| 100%                | Transfer learning with MobileNetV2 |

## Conclusions

- **Performance**: The pre-trained MobileNetV2 model outperformed both the baseline and convolutional models, demonstrating the effectiveness of transfer learning in image classification tasks.
  
- **Complexity vs. Performance**: While the convolutional model improved over the baseline, the pre-trained model achieved the highest accuracy with fewer training epochs and parameters.

- **Practical Application**: For tasks with limited training data, leveraging pre-trained models can significantly enhance model performance and reduce the risk of overfitting.
