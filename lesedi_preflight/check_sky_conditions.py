import requests
import datetime
from PIL import Image, ImageOps, ImageDraw
from io import BytesIO
from keras.models import load_model
import pytesseract
from astropy.time import Time
from astropy import units as u
import numpy as np

ATLAS_STH_ALLSKY_URL = "https://www.fallingstar.com/weather/sth/latest_clr400.jpg"

current_date_and_time = datetime.datetime.now()


def download_atlas_allsky_image(url=ATLAS_STH_ALLSKY_URL):
    """
    Downloads the latest AllSky image from the provided ATLAS allsky image URL and returns it as a PIL Image object.

    Parameters:
    url (str): The URL of the allsky image from the ATLAS Sutherland site status page to be downloaded.

    Returns:
    PIL.Image.Image or None: The downloaded image as a PIL Image object if successful,
                        otherwise None.
    """
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            allsky_img = Image.open(BytesIO(response.content))
            return allsky_img
        else:
            print("CSC - Failed to download image: Status code", response.status_code)
            return None
    except Exception as e:
        print("CSC - Error occurred while downloading image:", str(e))
        return None


def is_image_recent(allsky_img):
    """
    Check if the given image is recent based on the MJD extracted from the allsky image.

    Parameters:
        allsky_img (PIL.Image): The input image containing the MJD information.

    Returns:
        bool:
            - True if the image is recent (MJD is within the last 45 minutes).
            - False if the image is not recent (MJD is older than 45 minutes).
            - False if an error occurs during processing.
    """
    try:
        allsky_img_width, allsky_img_height = allsky_img.size

        # Crop the bottom left corner of the image to extract MJD information
        bottom_left = allsky_img.crop(
            (0, allsky_img_height - 30, 250, allsky_img_height)
        )

        ocr_text = pytesseract.image_to_string(bottom_left)

        # Extract the MJD value from the OCR text
        allsky_image_mjd = Time(float(ocr_text.split()[1]), format="mjd")

        # Calculate the stale cutoff time (45 minutes ago)
        stale_cutoff_time = Time.now() - 45 * u.minute

        # Check if the image time is later than the stale cutoff time
        if allsky_image_mjd > stale_cutoff_time:
            return True
        else:
            return False

    except Exception as e:
        print("CSC - Error occurred while checking if image is stale:", str(e))
        return False


def classify_sky_conditions(allsky_img):
    """
    Classify the sky conditions of an image using a pretrained Machine Learning Model.

    Parameters:
        allsky_img (PIL.Image): Image containing the allsky image.

    Returns:
        tuple: A tuple containing the predicted sky condition and its confidence score.
               If an error occurs during classification, returns None.
    """
    try:
        # Load the pre-trained classifier model
        classifier_model = load_model(
            "/Users/enzo/lesedi_preflight/machine_learning_cloud_cover/keras_model.h5",
            compile=False,
        )

        # Load class names
        with open(
            "/Users/enzo/lesedi_preflight/machine_learning_cloud_cover/labels.txt", "r"
        ) as f:
            class_names = [line.strip() for line in f.readlines()]

        # Prepare image for classification
        target_size = (224, 224)
        resized_image = ImageOps.fit(allsky_img, target_size, Image.Resampling.LANCZOS)
        image_array = np.asarray(resized_image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Classify the images sky condition
        prediction = classifier_model.predict(data)
        index = np.argmax(prediction)
        predicted_class = class_names[index]
        confidence_score = prediction[0][index]

        # Return predicted sky condition and confidence score
        return predicted_class[2:], confidence_score

    except Exception as e:
        print("CSC - Error occurred while ML model classified image:", str(e))
        return None


def annotate_and_save_sky_image(
    allsky_img,
    predicted_class,
    confidence_score,
    output_path="latest_classified_allsky_image.jpg",
):
    """
    Annotate the allsky image with classification results and save it.

    Parameters:
        allsky_img (PIL.Image): Image containing the allsky image.
        predicted_class (str): Predicted class of the sky condition.
        confidence_score (float): Confidence score of the prediction.
        output_path (str): Path to save the annotated image (default is 'latest_classified_allsky_image.jpg').

    Returns:
        None.
    """
    try:
        # Create a drawing context
        draw = ImageDraw.Draw(allsky_img)

        # Get the current Modified Julian Date (MJD)
        mjd = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Compose the text to be annotated
        text = (
            f"LESEDI AUTOMATED PREFLIGHT\n"
            f"MJD: {mjd}\n"
            f"Predicted: {predicted_class}\n"
            f"Confidence Score: {confidence_score:.2f}"
        )

        # Annotate the image with the classification results
        draw.text((10, 10), text, fill="red")

        # Save the annotated image
        allsky_img.save(output_path, quality=100)
        print(f"Annotated image saved to {output_path}")

        

    except Exception as e:
        # Handle exceptions and print an error message
        print(f"CSC - Error occurred while annotating and saving the allsky image: {str(e)}")
       
