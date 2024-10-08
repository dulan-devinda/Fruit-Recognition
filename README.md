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
After increasing the epochs to 20, the model started to converge better, showing significant improvement in both training and validation accuracy.

### Results
- **Training Accuracy**: (Add percentage here)
- **Validation Accuracy**: (Add percentage here)

#### Accuracy and Loss Graphs:
![Accuracy After 20 Epochs](path_to_accuracy_chart_20_epochs.png)

#### Confusion Matrix:
![Confusion Matrix After 20 Epochs](path_to_confusion_matrix_20_epochs.png)
## Experiment 3: Changed Model Architecture

### Description
The model architecture was modified by adding more convolutional layers and using dropout for regularization. This improved the model's ability to generalize and reduced overfitting.

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
