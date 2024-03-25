import cv2
import numpy as np


def analyse_cloud_cover(image_name):
    # Load the image
    image = cv2.imread(image_name)

    # Get image dimensions
    height, width = image.shape[:2]

    # Define the diameter of the circular ROI (2/3 of the width)
    roi_diameter = int(5/6 * width)
    roi_radius = roi_diameter // 2

    # Define the center of the circular ROI
    center_x = width // 2
    center_y = height // 2

    # Create a black background image with the same dimensions as the original image
    overlay = np.zeros_like(image)

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a circular mask on the HSV image
    hsv_roi = np.zeros_like(hsv)
    cv2.circle(hsv_roi, (center_x, center_y), roi_radius, (255, 255, 255), thickness=-1)

    # Threshold the HSV ROI to get only the sky region
    lower_sky = np.array([90, 50, 50])
    upper_sky = np.array([130, 255, 255])
    roi_mask = cv2.inRange(hsv, lower_sky, upper_sky)

    # Count non-zero pixels in the mask
    sky_pixels = cv2.countNonZero(roi_mask)

    # Calculate the total number of pixels in the ROI
    roi_total_pixels = np.count_nonzero(hsv_roi)

    # Calculate the percentage of sky pixels within the ROI
    sky_percentage = (sky_pixels / roi_total_pixels) * 100

    # Define a threshold for cloud cover percentage
    cloud_threshold = 40  # You may adjust this threshold as needed

    # Determine if there is cloud cover based on the percentage of sky pixels within the ROI
    if sky_percentage > cloud_threshold:
        text_color = (0, 255, 0)  # Green color for no clouds
        ring_color = (0, 255, 0)  # Green color for no clouds
        cloud_status = "No Cloud Warning"
    else:
        text_color = (0, 0, 255)  # Red color for clouds
        ring_color = (0, 0, 255)  # Red color for clouds
        cloud_status = "Cloud Warning"

    # Draw the circular ring on the overlay
    cv2.circle(overlay, (center_x, center_y), roi_radius, ring_color, thickness=2)

    # Blend the overlay with the original image
    alpha = 0.5  # Overlay transparency (adjust as needed)
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Write the cloud status on the image
    text = "Sky Percentage: {:.2f}% - {}".format(sky_percentage, cloud_status)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, text, (center_x - 150, center_y + roi_radius + 50), font, 0.7, text_color, 2)

    # Display the result
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

analyse_cloud_cover('test_images/Cloud cover at 2024-03-24 16:00:01.975691: 18.0%.jpg')