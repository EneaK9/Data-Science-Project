![Polis Logo](/assets/logopolis.jpg)

Data Science Project
Water Bottles Classifier 

1) Define at least 3 classes of bottles of water (i.e. 3 different
brands of water)
2) Take for each class at least 100 photos (the more photos
the best will be the results, suggested 300) of different
formats (e.g. 500 ml, 1 l, 1,5 l, ...)
3) Build three Neural Network Classifiers:
1) Only Dense layers (baseline 90% validation accuracy)
2) Dense and Convolutional layers (baseline 93%
validation accuracy)
3) Use already-trained neural network (baseline 95%
validation accuracy)
4) Build three decision tree classifiers: CART, C 5.0
and Random Forest (minimum 100 trees)
5) Build a Naive Bayes classifier



 Group: Enea Kuca, Irdi Dona, Algert Kashari


Water Bottle Image Classification Project

Description
This project is centered around developing a robust image classification system capable of accurately categorizing images of water bottles into three distinct classes using Convolutional Neural Networks (CNN). The motivation behind this project stems from the growing demand for automated visual recognition systems in various industries, including manufacturing, retail, and environmental monitoring.
Objectives

1.	Accurate Classification: The primary objective is to achieve high accuracy in identifying and classifying different types of water bottles based on their visual attributes captured in images. This involves training machine learning models to distinguish between categories such as "Lajthiza," "Spring," and "Tepelena," which represent specific brands or types of water bottles.

2.	Model Evaluation: Evaluate and compare the performance of three distinct classifiers:
o	Baseline Model (Dense Layers Only): A straightforward architecture using densely connected layers for classification.
o	Convolutional Model: Incorporating convolutional layers to extract spatial features from images, which is effective for capturing patterns and structures in visual data.
o	Pre-trained Model (MobileNetV2): Utilizing a pre-trained MobileNetV2 model, leveraging transfer learning to capitalize on patterns learned from a large-scale dataset (ImageNet) and fine-tuning it for the specific task of water bottle classification.

3.	Mitigating Overfitting: Implement strategies such as early stopping and dropout regularization to prevent overfitting. Early stopping halts training when the model's performance on a validation set no longer improves, ensuring the model generalizes well to unseen data. Dropout regularization introduces randomness by temporarily removing a fraction of neurons during training, reducing the model's reliance on specific features and enhancing its ability to generalize.

4.	Visualizing Overfitting: Provide visual examples of overfitting scenarios through diagrams that illustrate the divergence between training and validation accuracies over epochs. These diagrams serve to highlight the importance of regularization techniques in maintaining model performance on new data.
Expected Outcomes
By the end of this project, we aim to:
•	Deploy a trained model capable of accurately classifying water bottle images with high validation accuracy.
•	Evaluate the trade-offs between model complexity, training efficiency, and performance across different classifiers.
•	Demonstrate the practical applicability of leveraging pre-trained models in scenarios with limited labeled data, showcasing their potential to enhance classification accuracy and reduce development time.
This project not only contributes to advancements in computer vision and machine learning applications but also addresses practical challenges in automated product recognition and quality control in manufacturing and retail environments.

Classifiers Overview
1.	Baseline Model (Dense Layers Only):
o	The baseline model employs a straightforward architecture consisting solely of densely connected layers, also known as fully connected layers.
o	It takes flattened input images and passes them through several dense layers with ReLU activation functions, which help in learning non-linear mappings between input features and class labels.
o	This model serves as a fundamental benchmark, providing a baseline validation accuracy of 90%.
o	To mitigate overfitting, dropout regularization with a rate of 0.5 is applied, which randomly drops half of the neurons during training to prevent the model from memorizing noise in the training data.

2.	Convolutional Model:
o	The convolutional model incorporates convolutional layers, which are pivotal in capturing spatial hierarchies of features from images.
o	It begins with convolutional layers that apply filters to extract features such as edges, textures, and patterns from the input images.
o	Each convolutional layer is followed by a max-pooling layer to downsample the feature maps, reducing the computational complexity while retaining important features.
o	This architecture is well-suited for image classification tasks, as it learns hierarchical representations of the input data.
o	The convolutional model achieves a significantly improved validation accuracy of 93%, demonstrating the efficacy of convolutional neural networks in extracting meaningful features from water bottle images.

3.	Pre-trained Model (MobileNetV2):
o	The pre-trained MobileNetV2 model utilizes transfer learning, leveraging knowledge from a model pre-trained on the large-scale ImageNet dataset.
o	MobileNetV2 is chosen for its lightweight architecture and high performance in mobile and embedded vision applications.
o	In this approach, the MobileNetV2 model is used as a feature extractor, where its convolutional base is frozen to prevent further training and preserve the learned features.
o	Global average pooling is applied to reduce the spatial dimensions of the feature maps produced by MobileNetV2, followed by dense layers with softmax activation for classification.
o	By fine-tuning the pre-trained model on the specific task of water bottle classification, this approach achieves the highest validation accuracy of 95%.
o	This demonstrates the effectiveness of transfer learning in adapting pre-existing knowledge to solve a new classification task efficiently.

Comparison Between Different Classifiers
•	Baseline Dense Model: Achieves a validation accuracy of 92.27%, relying solely on dense layers without leveraging spatial relationships in the input images.
•	Convolutional Model: Achieves a validation accuracy of 99.48%, showcasing the benefit of convolutional layers in learning spatial features and improving classification performance.
•	Pre-trained MobileNetV2: Attains a validation accuracy of 100%, surpassing both the baseline and convolutional models. This highlights the superior performance achieved through transfer learning, where the model leverages pre-existing knowledge to excel in image classification tasks.
•	Handling Overfitting and Underfitting
Resolution Strategies
1.	Early Stopping:
o	Overfitting occurs when a model learns to memorize the training data rather than generalize to unseen data. Early stopping is a technique used to mitigate overfitting by monitoring the model's performance on a validation set during training.
o	During training, if the validation loss stops improving or begins to degrade after a certain number of epochs, early stopping halts the training process to prevent further overfitting.
o	In our project, early stopping is implemented by monitoring the validation loss. When no improvement is observed over a specified number of epochs (patience), training is terminated early to retain the model's ability to generalize.
2.	Dropout Regularization:
o	Dropout regularization is a technique used to reduce overfitting by randomly dropping a fraction of neurons (along with their connections) during training.
o	By randomly deactivating neurons, dropout introduces noise into the learning process, forcing the model to learn redundant representations and reducing reliance on specific features.
o	In our models, dropout layers with dropout rates of 0.5 and 0.3 are strategically inserted after dense layers to regularize the learning process.
o	This technique enhances the model's ability to generalize by making it more robust and less prone to memorizing noise in the training data.


Examples of Diagrams Showing Overfitting
1.	Training and Validation Accuracy Plot:
o	Diagrams illustrating overfitting typically depict the divergence between training accuracy and validation accuracy over epochs.
o	In an overfitting scenario, the training accuracy continues to improve as the model learns to fit the training data more closely.
o	Meanwhile, the validation accuracy either plateaus or even starts to decrease after reaching an optimal point, indicating that the model's performance on unseen data is not improving.
o	These diagrams serve as visual indicators of overfitting, highlighting the importance of implementing regularization techniques like dropout and early stopping to maintain model performance on new data.

Comparison Between Different Classifiers
•	Baseline Dense Model: While achieving a respectable validation accuracy of 92.27%, it is more susceptible to overfitting due to its reliance on densely connected layers without capturing spatial relationships in images.
•	Convolutional Model: With an impressive validation accuracy of 99.48%, the convolutional model demonstrates effective feature extraction through convolutional and pooling layers, reducing overfitting by learning hierarchical representations.
•	Pre-trained MobileNetV2: The pre-trained model achieves the highest validation accuracy of 100% by leveraging transfer learning, which effectively mitigates overfitting by adapting knowledge from a large-scale dataset (ImageNet) to the specific task of water bottle classification.
.


Comparison Between Different Classifiers

Classifier	            Validation Accuracy	      Key Features
Baseline Dense	           92.27%	                 Dense layers only
Convolutional	           99.48%	                 Convolutional layers added
Pre-trained MobileNet	   100%	               Transfer learning with MobileNetV2

Conclusions
The water bottle image classification project has explored various models and strategies to achieve high accuracy in distinguishing between different types of water bottles. Through the evaluation of three distinct classifiers—baseline dense, convolutional, and pre-trained MobileNetV2—we have gained valuable insights into their performance and applicability.
Performance Evaluation:
•	Baseline Dense Model: Starting with a baseline validation accuracy of 90%, the model demonstrates a solid foundation for classification tasks using densely connected layers. However, its limitation lies in its vulnerability to overfitting due to the lack of feature extraction capabilities inherent in convolutional architectures.
•	Convolutional Model: Incorporating convolutional layers for feature extraction significantly improves performance, achieving a validation accuracy of 93%. This model captures spatial hierarchies in images, making it more robust against overfitting compared to the baseline dense model.
•	Pre-trained MobileNetV2: Leveraging transfer learning from the pre-trained MobileNetV2 architecture yields the highest validation accuracy of 95%. By adapting knowledge learned from the ImageNet dataset, the model effectively learns relevant features for water bottle classification while mitigating overfitting. This underscores the efficacy of transfer learning in optimizing model performance with limited labeled data.
Complexity vs. Performance Trade-off:
•	Model Complexity: As expected, the complexity of the models increases with their capability to learn intricate features from data. The baseline dense model, being the simplest, is outperformed by the more sophisticated convolutional and pre-trained models in terms of accuracy.
•	Performance: The pre-trained MobileNetV2 model emerges as the top performer, achieving validation accuracy of 95%. It strikes a balance between model complexity and performance, demonstrating the effectiveness of leveraging pre-trained models for image classification tasks.
Practical Implications:
•	Transfer Learning Benefits: For practical applications with limited annotated data, adopting pre-trained models such as MobileNetV2 proves advantageous. It not only accelerates model development but also enhances classification accuracy while minimizing the risk of overfitting.
•	Generalization Capability: By addressing overfitting through techniques like dropout regularization and early stopping, the models demonstrate improved generalization ability. They maintain robust performance on unseen data, crucial for real-world deployment scenarios.
Future Directions:
•	Fine-tuning and Ensemble Approaches: Further improving model performance can be explored through fine-tuning of pre-trained models on specific water bottle datasets. Ensemble methods could also be investigated to combine predictions from multiple models for enhanced accuracy.
•	Data Augmentation: Augmenting the dataset with synthetic data can potentially increase model robustness and reduce overfitting, especially in scenarios with limited training samples.
•	Deployment and Scaling: Considerations for deploying the model in resource-constrained environments and scaling to larger datasets should be addressed for broader applicability.
In conclusion, the water bottle image classification project highlights the significance of model selection, regularization techniques, and transfer learning in achieving high accuracy and mitigating overfitting. These findings provide a foundation for future research and practical implementations in image recognition and classification domains.
