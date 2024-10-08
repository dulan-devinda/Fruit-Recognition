# Fruit-Recognition CNN

## Project Overview
This project implements a fruit recognition system using CNNs. The model was trained on a large dataset with over 100 fruit classes. Various architectures and configurations were tested to improve accuracy and performance.
## Experiment 1: Initial Training with 6 Epochs

### Description
The model was initially trained for 6 epochs. The training and validation accuracy were recorded, but the model showed signs of underfitting, indicating that more epochs were needed.

### Results
- **Test Accuracy**: 0.94
### Image Prediction Example

You can load an image and predict the fruit class using the model. Below is an example of how to use the model to classify an image of a fruit.

```python
import numpy as np
from tensorflow.keras.preprocessing import image

# Load and preprocess a new image
img_path = '/content/drive/MyDrive/Fruit Recognition Dataset/bunch-bananas-isolated-on-white-260nw-1722111529.webp'
img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Rescale

# Make a prediction
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the class index

# Get class names from the training generator
class_names = list(train_generator.class_indices.keys())  # This gives you the fruit names
predicted_class_name = class_names[predicted_class_index]

print(f'Predicted Class Index: {predicted_class_index}')
print(f'Predicted Class Name: {predicted_class_name}')
```
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step
Predicted Class Index: 106
Predicted Class Name: Physalis with Husk 1
```

#### Accuracy and Loss Graphs:
![Accuracy After 6 Epochs](https://github.com/dulan-devinda/Fruit-Recognition/blob/main/Images/graphs%20for%206%20epochs.png?raw=true)

## Experiment 2: Increased Epochs to 20

### Description
In this experiment, we increased the number of training epochs from 6 to 20 to allow the model more time to learn from the dataset. By extending the training process, the model began to converge better, showing significant improvements in both training and validation accuracy. Additionally, a T4 GPU was used for the training process, which drastically reduced training time compared to using a CPU, making it feasible to run more epochs in a reasonable time.

### Results
- **Test Accuracy**: 0.96
### Image Prediction Example

You can use the trained model to predict the class of fruits from both the test set of the dataset and new images. Below are examples that demonstrate the model's behavior on each.

#### Example 1: Image from the Test Set (Strawberry)
In this example, the image is from the test set of the dataset, and the model correctly identifies it.

```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess a test image from the dataset
img_path = '/content/dataset/fruits-360_dataset_100x100/fruits-360/Test/Strawberry 1/325_100.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(100, 100))  # Resize image to the size your model expects
img_array = image.img_to_array(img)  # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch (1, 100, 100, 3)

# Normalize the image (optional depending on how your training data was preprocessed)
img_array = img_array / 255.0

# Make a prediction
predictions = model.predict(img_array)

# Get the predicted class index and name
predicted_class_index = np.argmax(predictions, axis=1)[0]  # Index of the predicted class
predicted_class_name = class_names[predicted_class_index]  # Get the class name

print(f'Predicted Class Index: {predicted_class_index}')
print(f'Predicted Class Name: {predicted_class_name}')
```
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
Predicted Class Index: 124
Predicted Class Name: Strawberry 1
```
Explanation:

The model successfully predicts the class as Strawberry 1 when tested on an image from the dataset's test folder. This demonstrates that the model performs well on images it has seen similar patterns of during training.

#### Example 2: New Image (Banana)
In this case, we test the model with a new image (not from the dataset), and the model misclassifies the image.
```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess a new image (not in the dataset)
img_path = '/content/drive/MyDrive/Fruit Recognition Dataset/banana1.jfif'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(100, 100))  # Resize image to the size your model expects
img_array = image.img_to_array(img)  # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch (1, 100, 100, 3)

# Normalize the image (optional depending on how your training data was preprocessed)
img_array = img_array / 255.0

# Make a prediction
predictions = model.predict(img_array)

# Get the predicted class index and name
predicted_class_index = np.argmax(predictions, axis=1)[0]  # Index of the predicted class
predicted_class_name = class_names[predicted_class_index]  # Get the class name

print(f'Predicted Class Index: {predicted_class_index}')
print(f'Predicted Class Name: {predicted_class_name}')
```
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
Predicted Class Index: 106
Predicted Class Name: Physalis with Husk 1
```
Explanation:

The model misclassifies a banana image as Physalis with Husk 1. This shows that while the model performs well on the dataset images, it struggles to generalize to new, unseen images. This indicates that the model might need further fine-tuning or training on a more diverse dataset to improve its performance on real-world images.

#### Accuracy and Loss Graphs:
![Accuracy After 20 Epochs](https://github.com/dulan-devinda/Fruit-Recognition/blob/main/Images/graphs%20for%2020%20epochs.png?raw=true)


## Experiment 3: Changed Model Architecture

### Description
The model architecture was enhanced to improve its ability to generalize and reduce overfitting. The new architecture consists of three Conv2D layers, each followed by a MaxPooling2D layer, a Flatten layer, and two Dense layers, with one Dropout layer between them for regularization. The Conv2D layers use 32, 64, and 128 filters, respectively, and the model has approximately 3.4 million trainable parameters.

### Results
- **Training Accuracy**: (Add percentage here)
- **Validation Accuracy**: (Add percentage here)

#### Accuracy and Loss Graphs:
![Accuracy After Architecture Change](path_to_accuracy_chart_architecture_change.png)

#### Confusion Matrix:
![Confusion Matrix After Architecture Change](path_to_confusion_matrix_architecture_change.png)
## Final Model Performance

### Description
The final model achieved significant improvements after fine-tuning and increasing the number of epochs. The confusion matrix shows fewer misclassifications, and the accuracy has increased significantly.

### Results
- **Training Accuracy**: 100%
- **Validation Accuracy**: 97.8%

#### Final Accuracy and Loss Graphs:
![Final Accuracy](path_to_final_accuracy_chart.png)

#### Final Confusion Matrix:
![Final Confusion Matrix](path_to_final_confusion_matrix.png)
## Conclusion
By experimenting with different architectures and increasing the number of epochs, the model’s accuracy improved significantly. The final model can classify over 100 types of fruits with an accuracy of 97.8%. Data augmentation and regularization techniques played a key role in improving the model's generalization.

Future improvements may include fine-tuning with pre-trained models like ResNet or EfficientNet to further boost performance.
