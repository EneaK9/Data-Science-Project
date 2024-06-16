![Polis Logo](/assets/logopolis.jpg)

# Data Science Project

## Water Bottle Image Classification Project

# Description

The main goal of this research is to create a reliable image classification system that uses convolutional neural networks (CNN) to correctly classify photos of water bottles into three different groups. The increasing need for automated visual identification systems across a range of industries, such as manufacturing, retail, and environmental monitoring, is the driving force behind this initiative.

### Objectives

1. **Accurate Classification**:

   The main goal is to categorize and identify various types of water bottles with high accuracy using the visual characteristics that are collected in photos. In order to do this, machine learning models must be trained to differentiate between categories that represent different brands or varieties of water bottles, such as "Lajthiza," "Spring," and "Tepelena."

2. **Model Evaluation**:

   - **Evaluate and compare the performance of three distinct classifiers**:
   - **Baseline Model (Dense Layers Only)**:
     - A simple classification architecture with highly linked layers.
   - **Convolutional Model**:
     - Using convolutional layers to extract spatial characteristics from pictures is a useful method for identifying structures and patterns in visual data.
   - **Pre-trained Model (MobileNetV2)**:
     - Using a pre-trained MobileNetV2 model, optimizing it for the particular objective of classifying water bottles by using transfer learning to take use of patterns discovered from a large-scale dataset (ImageNet).

3. **Mitigating Overfitting**:

   - Use techniques like dropout regularization and early halting to avoid overfitting. In order to make sure the model generalizes adequately to new data, early stopping stops training when the model's performance on a validation set no longer increases. By momentarily eliminating a portion of neurons during training, dropout regularization adds unpredictability, lessening the model's dependence on certain features, and improving its generalization capacity.

4. **Visualizing Overfitting**:
   - Use diagrams that show how training and validation accuracy over epochs diverge to give visual examples of overfitting events. The significance of regularization strategies in preserving model performance on fresh data is illustrated by these graphics.

### Expected Outcomes

Our project's goals are to:
• Implement a trained model that can correctly identify photos of water bottles with a high validation accuracy by the end.
• Assess the trade-offs between performance across various classifiers, training efficiency, and model complexity.
• Showcase how using pre-trained models may be useful in situations when there isn't a lot of labeled data, since it can improve classification accuracy and speed up development.

    This research addresses real-world issues with automated product detection and quality control in industrial and retail settings, in addition to advancing computer vision and machine learning applications.

### Classifiers Overview

### Classifier 1: Baseline Model (Dense Layers Only)

#### Architecture

The neural network design of the baseline model is straightforward and consists just of dense (completely connected) layers. In order to evaluate the performance of increasingly complicated designs, this model acts as a fundamental benchmark.

- **Input Layer**: Flattens the 2D image input into a 1D array.
- **Hidden Layers**: Includes two dense layers with ReLU activation functions.
  - First Dense Layer: 128 neurons with ReLU activation.
  - Second Dense Layer: 64 neurons with ReLU activation.
- **Dropout Layers**: Added after each hidden layer to reduce overfitting by randomly setting a fraction of input units to zero during training.
  - Dropout rate: 0.5 after the first dense layer and 0.3 after the second dense layer.
- **Output Layer**: A dense layer with a softmax activation function to output class probabilities for the multi-class classification task.

#### Implementation

```python
# Building the model
model = Sequential([
    Flatten(input_shape=(50, 50, 1)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax'),

])

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

#### Performance

Convolutional layers, which can capture spatial hierarchies in pictures, are more suited for image data and are included in the convolutional model. Multiple convolutional layers, pooling layers, and dense layers for classification are the order in which this model is organized.

### Classifier 2: Convolutional Model

#### Architecture

The convolutional model incorporates convolutional layers which are better suited for image data as they can capture spatial hierarchies in images. This model includes multiple convolutional layers followed by pooling layers, then dense layers for classification.

- **Convolutional Layers**: Extract spatial features from the images.
- **Conv2D Layer 1**: 32 filters, kernel size 3x3, ReLU activation.
- **MaxPooling2D Layer 1**: Pool size 2x2.
- **Conv2D Layer 2**: 64 filters, kernel size 3x3, ReLU activation.
- **MaxPooling2D Layer 2**: Pool size 2x2.
- **Conv2D Layer 3**: 128 filters, kernel size 3x3, ReLU activation.
- **MaxPooling2D Layer 3**: Pool size 2x2.
- **Flatten Layer**: Flattens the 3D feature maps to 1D feature vectors.
- **Dense Layers**: Perform final classification.
- **Dense Layer**: 512 neurons with ReLU activation.
- **Dropout Layer**: 0.5 dropout rate for regularization.
- **Output Layer**: Dense layer with softmax activation.

#### Implementation

```python
# Building the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```

#### Performance

Training Accuracy: Improvement in feature extraction has led to a higher training accuracy compared to the baseline model.
Validation Accuracy: 93%, demonstrating enhanced generalization and demonstrating how well convolutional layers capture characteristics from images..

### Classifier 3: Pre-trained Model (MobileNetV2)

#### Architecture

Using the MobileNetV2 architecture, which has been pre-trained on a sizable dataset (ImageNet), this model makes use of transfer learning. Utilizing pre-learned characteristics from a sizable and varied dataset is made easier with transfer learning.

- **Base Model**: Without the upper classification layer, MobileNetV2.
  - A feature extractor is based on the base model.
  - Pre-trained weights are retained by freezing the layers.
- **Custom Top Layers**: Added for the specific classification task.
  - _GlobalAveragePooling2D Layer_: Reduces each feature map to a single value.
  - _Dense Layer_: 512 neurons with ReLU activation.
  - _Dropout Layer_: 0.5 dropout rate for regularization.
  - _Output Layer_: Dense layer with softmax activation.

#### Implementation

```python
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freezing the base model
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```

#### Performance

- Training Accuracy: High training accuracy because the model makes use of previously learnt characteristics.
- Validation Accuracy: 95%, demonstrating good generalization and utilizing transfer learning to classify images.

### Comparison Between Different Classifiers

    • Baseline Dense Model: O3nly uses dense layers and ignores spatial connections in the input pictures, achieves a validation accuracy of 92.27%.
    • Convolutional Model: Showcases the value of convolutional layers in learning spatial information and enhancing classification performance, achieving a validation accuracy of 99.48%.
    • Pre-trained MobileNetV2: Outperforms the baseline and convolutional models, achieving 100% validation accuracy. This demonstrates the improved performance attained via transfer learning, in which the model performs better on picture classification tasks by utilizing prior information.

## Handling Overfitting and Underfitting

### Resolution Strategies

1. **Early Stopping**:
   - When a model learns too much from the training set without properly generalizing to new data, it is said to be overfitting. By keeping an eye on the model's performance on a validation set during training, early halting helps to alleviate this problem.
   - Early stopping interrupts the training process to stop additional overfitting if, after a certain number of epochs, the validation loss stops improving or starts to deteriorate.
   - We use validation loss monitoring to implement early halting in our project. To preserve the model's generalizability, training is stopped early if no improvement is shown after a certain number of epochs (patience).
2. **Dropout Regularization**:
   - During training, a portion of the neurons and their connections are randomly dropped as part of a process called dropout regularization, which aims to minimize overfitting.
   - Dropout adds chaos to the learning process by killing neurons at random, which forces the model to acquire redundant representations and lessens dependence on certain information.
   - To regularize the learning process, dropout layers with dropout rates of 0.5 and 0.3 are placed after dense layers in our models. By strengthening the model and decreasing its susceptibility to remembering noise in the training set, this method improves the model's capacity for generalization.

### Examples of Diagrams Showing Overfitting

- ![Overfitting Diagram Dense Layer](/assets/dense-layer.png)

- ![Overfitting Diagram CNN](/assets/cnn-diagram.png)

- ![Overfitting Diagram Pre-trained](/assets/pre-trained.png)

The diagram depicts a classic example of overfitting, characterized by the phenomenon where training accuracy steadily improves while validation accuracy either plateaus or deteriorates. This discrepancy highlights the model's tendency to excessively fit to the training data, potentially losing its ability to generalize effectively to unseen data.

## Comparison Between Different Classifiers

| Classifier            | Validation Accuracy | Key Features                       |
| --------------------- | ------------------- | ---------------------------------- |
| Baseline Dense        | 92.27%%             | Dense layers only                  |
| Convolutional         | 99.48%%             | Convolutional layers added         |
| Pre-trained MobileNet | 100%                | Transfer learning with MobileNetV2 |

## Conclusions

- **Performance**: The pre-trained MobileNetV2 model surpassed both the baseline and convolutional models, highlighting the efficacy of transfer learning in optimizing image classification tasks.
- **Complexity vs. Performance**: Although the convolutional model exhibited improvement over the baseline, the pre-trained model achieved superior accuracy while requiring fewer training epochs and parameters.

- **Practical Application**: In scenarios where training data is scarce, integrating pre-trained models can notably elevate model accuracy and mitigate concerns related to overfitting.

## Contributors

### Enea Kuca

### Algert Kashari

### Irdi Dona
