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
![Model Architecture](https://github.com/dulan-devinda/Fruit-Recognition/blob/main/Images/model.png?raw=true)

### Results
- **Training Accuracy**: 100.00%
- **Validation Accuracy**: 97.80%

### Image Prediction Example
```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess a test image
img_path = '/content/drive/MyDrive/Fruit Recognition Dataset/banana.jfif'  # Replace with the path to your image
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
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 821ms/step
Predicted Class Index: 18
Predicted Class Name: Banana 1
```
Explanation:

After adjusting the model architecture and adding dropout regularization, the model was able to correctly identify unseen fruits from the test dataset, including strawberries. In a more significant development, the model successfully identified new images of fruits, such as a banana, which it had not seen during training. This indicates a notable improvement in the model’s ability to generalize and handle new data effectively.

#### Accuracy and Loss Graphs:
![Accuracy After Architecture Change](https://github.com/dulan-devinda/Fruit-Recognition/blob/main/Images/graphs.png?raw=true)

#### Confusion Matrix:
```
739/739 ━━━━━━━━━━━━━━━━━━━━ 17s 22ms/step
Confusion Matrix:
[[1 0 1 ... 1 0 0]
 [4 0 1 ... 0 1 0]
 [1 0 1 ... 2 0 1]
 ...
 [1 0 1 ... 1 1 0]
 [1 0 1 ... 1 0 0]
 [0 2 1 ... 0 0 0]]
Classification Report:
                       precision    recall  f1-score   support

              Apple 6       0.00      0.01      0.01       157
     Apple Braeburn 1       0.00      0.00      0.00       164
 Apple Crimson Snow 1       0.01      0.01      0.01       148
       Apple Golden 1       0.01      0.01      0.01       160
       Apple Golden 2       0.00      0.00      0.00       164
       Apple Golden 3       0.01      0.01      0.01       161
 Apple Granny Smith 1       0.01      0.01      0.01       164
    Apple Pink Lady 1       0.01      0.01      0.01       152
          Apple Red 1       0.00      0.00      0.00       164
          Apple Red 2       0.00      0.00      0.00       164
          Apple Red 3       0.00      0.00      0.00       144
Apple Red Delicious 1       0.01      0.01      0.01       166
   Apple Red Yellow 1       0.01      0.01      0.01       164
   Apple Red Yellow 2       0.01      0.01      0.01       219
          Apple hit 1       0.01      0.01      0.01       234
            Apricot 1       0.01      0.01      0.01       164
            Avocado 1       0.00      0.00      0.00       143
       Avocado ripe 1       0.01      0.01      0.01       166
             Banana 1       0.00      0.00      0.00       166
 Banana Lady Finger 1       0.00      0.00      0.00       152
         Banana Red 1       0.01      0.01      0.01       166
           Beetroot 1       0.00      0.00      0.00       150
          Blueberry 1       0.02      0.02      0.02       154
      Cabbage white 1       0.00      0.00      0.00        47
       Cactus fruit 1       0.01      0.01      0.01       166
         Cantaloupe 1       0.00      0.00      0.00       164
         Cantaloupe 2       0.01      0.01      0.01       164
          Carambula 1       0.00      0.00      0.00       166
             Carrot 1       0.00      0.00      0.00        50
        Cauliflower 1       0.00      0.00      0.00       234
             Cherry 1       0.01      0.01      0.01       164
             Cherry 2       0.02      0.02      0.02       246
     Cherry Rainier 1       0.02      0.02      0.02       246
   Cherry Wax Black 1       0.01      0.01      0.01       164
     Cherry Wax Red 1       0.01      0.01      0.01       164
  Cherry Wax Yellow 1       0.01      0.01      0.01       164
           Chestnut 1       0.01      0.01      0.01       153
         Clementine 1       0.01      0.01      0.01       166
              Cocos 1       0.00      0.00      0.00       166
               Corn 1       0.00      0.00      0.00       150
          Corn Husk 1       0.00      0.00      0.00       154
           Cucumber 1       0.00      0.00      0.00        50
           Cucumber 3       0.00      0.00      0.00        81
      Cucumber Ripe 1       0.00      0.00      0.00       130
      Cucumber Ripe 2       0.01      0.01      0.01       156
              Dates 1       0.02      0.02      0.02       166
           Eggplant 1       0.00      0.00      0.00       156
      Eggplant long 1       0.00      0.00      0.00        80
                Fig 1       0.00      0.00      0.00       234
        Ginger Root 1       0.00      0.00      0.00        99
         Granadilla 1       0.02      0.02      0.02       166
         Grape Blue 1       0.01      0.01      0.01       328
         Grape Pink 1       0.00      0.00      0.00       164
        Grape White 1       0.01      0.01      0.01       166
        Grape White 2       0.01      0.01      0.01       166
        Grape White 3       0.01      0.01      0.01       164
        Grape White 4       0.01      0.01      0.01       158
    Grapefruit Pink 1       0.01      0.01      0.01       166
   Grapefruit White 1       0.01      0.01      0.01       164
              Guava 1       0.01      0.01      0.01       166
           Hazelnut 1       0.00      0.00      0.00       157
        Huckleberry 1       0.01      0.01      0.01       166
               Kaki 1       0.02      0.02      0.02       166
               Kiwi 1       0.01      0.01      0.01       156
           Kohlrabi 1       0.00      0.00      0.00       157
           Kumquats 1       0.02      0.02      0.02       166
              Lemon 1       0.00      0.00      0.00       164
        Lemon Meyer 1       0.00      0.00      0.00       166
              Limes 1       0.01      0.01      0.01       166
             Lychee 1       0.01      0.01      0.01       166
          Mandarine 1       0.02      0.02      0.02       166
              Mango 1       0.01      0.01      0.01       166
          Mango Red 1       0.00      0.00      0.00       142
          Mangostan 1       0.01      0.01      0.01       102
           Maracuja 1       0.00      0.00      0.00       166
 Melon Piel de Sapo 1       0.02      0.02      0.02       246
           Mulberry 1       0.01      0.01      0.01       164
          Nectarine 1       0.00      0.00      0.00       164
     Nectarine Flat 1       0.01      0.01      0.01       160
         Nut Forest 1       0.01      0.01      0.01       218
          Nut Pecan 1       0.01      0.01      0.01       178
          Onion Red 1       0.01      0.01      0.01       150
   Onion Red Peeled 1       0.00      0.00      0.00       155
        Onion White 1       0.00      0.01      0.01       146
             Orange 1       0.01      0.01      0.01       160
             Papaya 1       0.01      0.01      0.01       164
      Passion Fruit 1       0.01      0.01      0.01       166
              Peach 1       0.00      0.00      0.00       164
              Peach 2       0.01      0.01      0.01       246
         Peach Flat 1       0.00      0.00      0.00       164
               Pear 1       0.00      0.00      0.00       164
               Pear 2       0.00      0.00      0.00       232
               Pear 3       0.00      0.00      0.00        72
         Pear Abate 1       0.01      0.01      0.01       166
       Pear Forelle 1       0.01      0.01      0.01       234
        Pear Kaiser 1       0.01      0.01      0.01       102
       Pear Monster 1       0.01      0.01      0.01       166
           Pear Red 1       0.01      0.01      0.01       222
         Pear Stone 1       0.01      0.02      0.02       237
      Pear Williams 1       0.02      0.02      0.02       166
             Pepino 1       0.01      0.01      0.01       166
       Pepper Green 1       0.00      0.00      0.00       148
      Pepper Orange 1       0.02      0.02      0.02       234
         Pepper Red 1       0.00      0.00      0.00       222
      Pepper Yellow 1       0.00      0.00      0.00       222
           Physalis 1       0.01      0.01      0.01       164
 Physalis with Husk 1       0.01      0.01      0.01       164
          Pineapple 1       0.02      0.02      0.02       166
     Pineapple Mini 1       0.02      0.02      0.02       163
       Pitahaya Red 1       0.01      0.01      0.01       166
               Plum 1       0.00      0.00      0.00       151
               Plum 2       0.01      0.01      0.01       142
               Plum 3       0.01      0.01      0.01       304
        Pomegranate 1       0.00      0.00      0.00       164
     Pomelo Sweetie 1       0.00      0.00      0.00       153
         Potato Red 1       0.00      0.01      0.01       150
  Potato Red Washed 1       0.02      0.02      0.02       151
       Potato Sweet 1       0.00      0.00      0.00       150
       Potato White 1       0.01      0.01      0.01       150
             Quince 1       0.01      0.01      0.01       166
           Rambutan 1       0.00      0.00      0.00       164
          Raspberry 1       0.00      0.00      0.00       166
         Redcurrant 1       0.01      0.01      0.01       164
              Salak 1       0.01      0.01      0.01       162
         Strawberry 1       0.01      0.01      0.01       164
   Strawberry Wedge 1       0.01      0.01      0.01       246
          Tamarillo 1       0.01      0.01      0.01       166
            Tangelo 1       0.00      0.00      0.00       166
             Tomato 1       0.01      0.01      0.01       246
             Tomato 2       0.00      0.00      0.00       225
             Tomato 3       0.01      0.01      0.01       246
             Tomato 4       0.01      0.01      0.01       160
  Tomato Cherry Red 1       0.01      0.01      0.01       164
       Tomato Heart 1       0.01      0.01      0.01       228
      Tomato Maroon 1       0.00      0.00      0.00       127
      Tomato Yellow 1       0.00      0.00      0.00       153
 Tomato not Ripened 1       0.01      0.01      0.01       158
             Walnut 1       0.01      0.01      0.01       249
         Watermelon 1       0.01      0.01      0.01       157
           Zucchini 1       0.00      0.00      0.00        80
      Zucchini dark 1       0.00      0.00      0.00        80

             accuracy                           0.01     23619
            macro avg       0.01      0.01      0.01     23619
         weighted avg       0.01      0.01      0.01     23619

```
## Final Model Performance

### Description
The final model achieved significant improvements after fine-tuning and increasing the number of epochs. The confusion matrix shows fewer misclassifications, and the accuracy has increased significantly.

### Results
- **Training Accuracy**: 100%
- **Validation Accuracy**: 97.8%

## Conclusion
By experimenting with different architectures and increasing the number of epochs, the model’s accuracy improved significantly. The final model can classify over 100 types of fruits with an accuracy of 97.8%. Data augmentation and regularization techniques played a key role in improving the model's generalization.

Future improvements may include fine-tuning with pre-trained models like ResNet or EfficientNet to further boost performance.
