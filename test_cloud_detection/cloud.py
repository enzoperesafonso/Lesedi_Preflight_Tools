import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image

## THIS WORKS WELL
image = cv2.imread('test_images/Cloud cover at 2024-03-24 16:00:01.975691: 18.0%.jpg')

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for the color of the sky (blue)
upper_sky = np.array([123, 169, 230])
lower_sky = np.array([32, 67, 148])

# Threshold the HSV image to get only the sky region
mask = cv2.inRange(hsv, lower_sky, upper_sky)

# Bitwise AND operation to apply the mask on the original image
result = cv2.bitwise_and(image, image, mask=mask)

# Convert the mask to 3-channel format for compatibility with the original image
mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Invert the mask for visualization
mask_inv = cv2.bitwise_not(mask)
mask_inv_3channel = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

# Add the inverted mask to the original image
overlay = cv2.addWeighted(image, 1, mask_inv_3channel, 1, 0)

# Plot the original image and the overlaid result
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title('Visible Sky Overlay')
plt.axis('off')

plt.show()
