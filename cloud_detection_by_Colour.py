import cv2
import numpy as np

# Load the image
image = cv2.imread('test_images/Cloud cover at 2024-03-21 16:00:01.828674: 1.6%.jpg')

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for the color of the sky (blue)
upper_sky = np.array([123, 169, 230])
lower_sky = np.array([32, 67, 148])

# Threshold the HSV image to get only the sky region
mask = cv2.inRange(hsv, lower_sky, upper_sky)

# Count non-zero pixels in the mask
sky_pixels = cv2.countNonZero(mask)

# Calculate the total number of pixels in the image
total_pixels = image.shape[0] * image.shape[1]

# Calculate the percentage of sky pixels
sky_percentage = (sky_pixels / total_pixels) * 100

# Define a threshold for cloud cover percentage
cloud_threshold = 40  # You may adjust this threshold as needed

# Determine if there is cloud cover based on the percentage of sky pixels
if sky_percentage < cloud_threshold:
    print("Cloud cover detected")
else:
    print("No cloud cover detected")

# Display the mask (for visualization)
cv2.imshow('Sky Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()