import os
import json
import shutil
import random
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

# Step 1: Load the annotations
annotations_file = 'LI13/result.json'

with open(annotations_file, 'r') as f:
    annotations = json.load(f)
# This code reads an annotations file in JSON format.
# The variable annotations_file specifies the path to the JSON file containing annotations.
# The file is opened in read mode, and its contents are loaded into the annotations variable using json.load().

# Step 2: Prepare the training and validation datasets
data_dir = 'LI13'
train_dir = 'train'
val_dir = 'val'
# These lines define directory paths for the dataset and training/validation data directories.
# The data_dir variable represents the directory where the images and annotations are stored.
# The train_dir and val_dir variables represent the paths to the directories where the training and validation images will be copied, respectively.

# Delete train and val directories if they exist
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(val_dir):
    shutil.rmtree(val_dir)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
# This code checks if the train_dir and val_dir directories exist.
# If they exist, they are deleted using shutil.rmtree() to remove any existing data.
# Then, the code creates empty directories using os.makedirs() for both the training and validation data.

images = annotations['images']
random.shuffle(images)  # Shuffle the images randomly
# These lines retrieve the list of images from the annotations data.
# The images are shuffled randomly using random.shuffle() to introduce randomness in the training and validation data splitting process.

# Split ratio for training and validation
train_ratio = 0.8
train_size = int(len(images) * train_ratio)

train_images = images[:train_size]
val_images = images[train_size:]
# These lines define the training and validation data split ratio. train_ratio specifies the proportion of images that will be used for training (80% in this case).
# The total number of images is multiplied by train_ratio to determine the number of images for training (train_size).
# The remaining images are assigned to validation (val_images).

# Create the class subdirectories in the train directory
class_dirs_train = {}
for category in annotations['categories']:
    class_name = category['name']
    class_dir_train = os.path.join(train_dir, class_name)
    os.makedirs(class_dir_train, exist_ok=True)
    class_dirs_train[category['id']] = class_dir_train
# These lines create subdirectories for each class label in the training directory.
# It iterates over the categories defined in the annotations and creates a subdirectory for each class label using os.makedirs().
# The subdirectory path is stored in the class_dirs_train dictionary using the category ID as the key.

# Move images to the respective class subdirectories in the train directory
for image in train_images:
    src_path = os.path.join(data_dir, image['file_name'])
    if os.path.exists(src_path):
        annotation_id = image['id']
        annotation = next((ann for ann in annotations['annotations'] if ann['image_id'] == annotation_id), None)
        if annotation is not None:
            class_id = annotation['category_id']
            dst_path = os.path.join(class_dirs_train[class_id], os.path.basename(image['file_name']))
            shutil.copyfile(src_path, dst_path)
# This code copies the images from the original dataset directory to their respective class subdirectories in the training directory.
# It iterates over each image in train_images and retrieves the source path of the image.
# The corresponding annotation is retrieved based on the image ID, and the class ID is obtained from the annotation.
# The destination path is determined based on the class ID and the image's filename.
# Finally, shutil.copyfile() is used to copy the image file from the source path to the destination path.

# Create the class subdirectories in the validation directory
class_dirs_val = {}
for category in annotations['categories']:
    class_name = category['name']
    class_dir_val = os.path.join(val_dir, class_name)
    os.makedirs(class_dir_val, exist_ok=True)
    class_dirs_val[category['id']] = class_dir_val
# These lines create subdirectories for each class label in the validation directory.
# The process is similar to the creation of class subdirectories in the training directory, but here it is done for the validation directory.

# Move images to the respective class subdirectories in the validation directory
for image in val_images:
    src_path = os.path.join(data_dir, image['file_name'])
    if os.path.exists(src_path):
        annotation_id = image['id']
        annotation = next((ann for ann in annotations['annotations'] if ann['image_id'] == annotation_id), None)
        if annotation is not None:
            class_id = annotation['category_id']
            dst_path = os.path.join(class_dirs_val[class_id], os.path.basename(image['file_name']))
            shutil.copyfile(src_path, dst_path)
# This code copies the images from the original dataset directory to their respective class subdirectories in the validation directory.
# It follows a similar process as the image copying in the training directory, but here it is done for the validation images.

# Step 3: Define the model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(annotations['categories']), activation='softmax'))
# These lines define the architecture of the model using the Sequential API from Keras.
# Convolutional (Conv2D), pooling (MaxPooling2D), flatten (Flatten), and fully connected (Dense) layers are added to the model.
# The specific configuration of the layers determines the model's architecture and its ability to learn features from input images.
# The last layer uses the 'softmax' activation function to output probabilities for each class label based on the number of categories in annotations.

# Step 4: Set up data generators for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,  # Add rotation augmentation
    width_shift_range=0.2,  # Add width shift augmentation
    height_shift_range=0.2,  # Add height shift augmentation
    fill_mode='nearest'  # Fill mode for data augmentation
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
# These lines create instances of ImageDataGenerator for data augmentation and preprocessing.
# The train_datagen object is configured with various augmentation options, such as rescaling, shear range, zoom range,
# horizontal flip, rotation range, width shift range, height shift range, and fill mode. The val_datagen object is only rescaling the validation data.

# Step 5: Compile the model with adjusted learning rate
learning_rate = 0.0001  # Adjust the learning rate
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# These lines set up the model for training. The learning rate is defined as 0.0001, and an Adam optimizer with the specified learning rate is used.
# The model is compiled with the categorical cross-entropy loss function and the accuracy metric.

# Step 6: Set up model checkpoint for saving the best model
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
# This code sets up a model checkpoint callback to save the best model during training.
# The ModelCheckpoint callback is configured to monitor the validation loss ('val_loss') and save only the best model (save_best_only=True) based on the validation loss.

# Step 7: Train the model with learning rate schedule
def learning_rate_schedule(epoch):
    if epoch < 5:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(learning_rate_schedule)
# This code defines a learning rate schedule function learning_rate_schedule(epoch).
# The learning rate schedule is used to adjust the learning rate during training. In this case, the learning rate is decreased exponentially after the fifth epoch.
# The function returns the learning rate based on the current epoch. The LearningRateScheduler callback is created with the learning rate schedule function.

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
# These lines create data generators for training and validation data.
# The train_generator is created using the train_datagen instance, specifying the training directory, target image size, batch size, and categorical class mode.
# The val_generator is created similarly for the validation directory.

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,  # Increase the number of epochs
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[checkpoint, lr_scheduler]
)
# This code performs the actual model training. The model.fit() function is called with the training and validation data generators,
# number of steps per epoch, number of epochs, validation data, number of validation steps, and the defined callbacks (checkpoint and lr_scheduler).
# The training progress and metrics are stored in the history object.

# Step 8: Calculate accuracy on the validation dataset
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
# This code creates a new validation data generator for evaluating the model's performance.
# It uses the val_datagen instance but with a batch size of 1 and disables shuffling of the data (shuffle=False).

y_true = val_generator.classes # retrieves the true class labels for the validation data.
y_pred = model.predict(val_generator) #  uses the trained model to predict class probabilities for the validation data.
y_pred = tf.argmax(y_pred, axis=1).numpy() # converts the predicted probabilities into predicted class labels.
# These lines collectively perform inference using the trained model on the validation data.
# The true class labels are obtained, the model predicts class probabilities, and then the predicted class labels are extracted for further evaluation.

accuracy = sum(y_true == y_pred) / len(y_true)
print('Validation Accuracy:', accuracy)
# These lines calculate the accuracy of the model on the validation data. I
# t compares the true class labels (y_true) with the predicted class labels (y_pred) and calculates the accuracy
# by summing the number of correct predictions (y_true == y_pred) and dividing it by the total number of predictions (len(y_true)).
# The result is the validation accuracy. The accuracy is then printed to the console.

# Save the model
model.save('trained_model.h5')