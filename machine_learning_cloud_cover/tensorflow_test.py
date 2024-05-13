import requests
import datetime
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

ATLAS_STH_ALLSKY_URL = "https://www.fallingstar.com/weather/sth/latest_clr400.jpg"

current_date_and_time = datetime.datetime.now()
try:
    response = requests.get(ATLAS_STH_ALLSKY_URL)
    if response.status_code == 200:
        with open('machine_learning_cloud_cover/latest.jpg', "wb") as f:
            f.write(response.content)
        print("{} Image downloaded successfully".format(current_date_and_time))
    else:
        print("Failed to download image: Status code", response.status_code)
except Exception as e:
    print("Error occurred while downloading image:", str(e))
    
    
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("/Users/enzo/lesedi_preflight/machine_learning_cloud_cover/keras_model.h5", compile=False)

# Load the labels
class_names = open("/Users/enzo/lesedi_preflight/machine_learning_cloud_cover/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("machine_learning_cloud_cover/latest.jpg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print(prediction)
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)