# PREZENTACIJA 3
# Overall, this code loads a pre-trained model, performs object detection on an input image,
# and visualizes the results by drawing bounding boxes and
# displaying the detected class label and confidence level on the image.
import cv2
import numpy as np
from tensorflow.keras.models import load_model
# The code imports the necessary libraries: cv2 for image processing, numpy for numerical operations,
# and load_model from tensorflow.keras.models to load the trained model.

# Load the trained model
model = load_model('trained_model.h5')
# The code loads the pre-trained model from the file named 'trained_model.h5' and assigns it to the variable model.

# Define the class labels
class_labels = ['F1 car', 'Football ball', 'iPhone']

# This line defines the class labels for the objects that the model can detect.
# In this case, it contains the labels 'F1 car', 'Football ball', and 'iPhone'.

# Load the test image
image_path = 'Test images/40.png' # 22, 25, 30
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# This code block specifies the path to the test image ('Test images/18.jpg'), reads the image using cv2.imread,
# and assigns it to the variable image. It also retrieves the height, width,
# and number of channels of the image using the shape attribute.

# Resize the image to match the size used during training
target_size = (224, 224)
resized_image = cv2.resize(image, target_size)

# The code sets the desired size for the input image (target_size) to (224, 224) pixels.
# It then resizes the image using cv2.resize to match the target_size and assigns it to the variable resized_image.

# Preprocess the image
preprocessed_image = resized_image.astype(np.float32) / 255.0
preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

# This code block performs preprocessing on the resized image. First, it converts the pixel values of resized_image to
# floating-point numbers in the range [0, 1] by dividing by 255.0.
# Then, it adds an extra dimension to the image using np.expand_dims to match the input shape required by the model.

# Perform object detection
predictions = model.predict(preprocessed_image)
class_index = np.argmax(predictions)
class_label = class_labels[class_index]
confidence = predictions[0][class_index]

# The code uses the model to make predictions on the preprocessed image (preprocessed_image).
# The predictions are stored in the predictions variable.
# The np.argmax function is used to find the index of the class with the highest prediction probability.
# The corresponding class label is retrieved from class_labels and assigned to class_label.
# The confidence level (prediction probability) for the detected class is stored in confidence.

# Calculate the bounding box coordinates
bbox = predictions[0][:2] * np.array([target_size[1], target_size[0]])
bbox = bbox.astype(int)

# This code calculates the top-left coordinates of the bounding box based on the first two values of the predictions.
# It scales the values by the width and height of the target_size using np.array and element-wise multiplication.
# The resulting coordinates are then converted to integers using astype(int) and assigned to the variable bbox.

# Calculate the bottom right coordinates of the bounding box
bbox_bottom_right = bbox[:2] + predictions[0][2:4] * np.array([target_size[1], target_size[0]])
bbox_bottom_right = bbox_bottom_right.astype(int)

# Similar to the previous code block, this line calculates the bottom-right coordinates of the bounding box by adding
# the top-left coordinates (bbox[:2]) to the last two values of the predictions scaled by the target_size.
# The resulting coordinates are also converted to integers and stored in bbox_bottom_right.

# Scale the bounding box coordinates back to the original image size
scale_factor = (image_width / target_size[1], image_height / target_size[0])
bbox = (bbox * scale_factor).astype(int)
bbox_bottom_right = (bbox_bottom_right * scale_factor).astype(int)

# This code calculates the scale factors for width and height based on the ratio between the original image size and the target_size.
# It then scales the bounding box coordinates (bbox and bbox_bottom_right) back to the original image size
# by multiplying them with the corresponding scale factors. The coordinates are converted to integers using astype(int).

# Draw the rectangle and confidence level on the image
cv2.rectangle(image, (bbox[0], bbox[1]), (bbox_bottom_right[0], bbox_bottom_right[1]), (0, 255, 0), 2)
cv2.putText(image, f'{class_label}: {confidence:.2f}', (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# These lines draw a rectangle on the image using the calculated top-left and bottom-right coordinates.
# The rectangle is visualized as a green color ((0, 255, 0)) with a line thickness of 2 pixels.
# Additionally, it puts text on the image displaying the detected class label and confidence level near the top-left
# corner of the bounding box. The text is displayed in green color ((0, 255, 0)) with a font scale of 0.9 and a line thickness of 2 pixels.

# Print the detected class and confidence level
print('Detected Class:', class_label)
print('Confidence:', confidence)

# These lines print the detected class label and confidence level to the console.

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# These lines display the annotated image in a window titled 'Image' using cv2.imshow.
# It waits for a key press with cv2.waitKey(0), and then closes all windows using cv2.destroyAllWindows().