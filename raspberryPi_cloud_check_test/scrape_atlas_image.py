import os
import requests
import io
import cv2
import numpy as np

ATLAS_STH_ALLSKY_URL = "https://www.fallingstar.com/weather/sth/latest_clr400.jpg"

def download_allsky_image():
    try:
        response = requests.get(ATLAS_STH_ALLSKY_URL)
        if response.status_code == 200:
            image_bytes = io.BytesIO(response.content)
            image_array = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
        else:
            print("Failed to download image: Status code", response.status_code)
            return None
    except Exception as e:
        print("Error occurred while downloading image:", str(e))
        return None

