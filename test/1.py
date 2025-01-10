import cv2
import numpy as np

# Load the image
image = cv2.imread('s1.PNG')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to get a binary image
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Find contours of the white spots
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate the total area of the image
total_area = image.shape[0] * image.shape[1]

# Calculate the area of the white spots
white_spots_area = sum(cv2.contourArea(contour) for contour in contours)

# Calculate the percentage of the area occupied by white spots
percentage_area = (white_spots_area / total_area) * 100

print(f'Total area: {total_area} pixels')
print(f'White spots area: {white_spots_area} pixels')
print(f'Percentage of area occupied by white spots: {percentage_area:.2f}%')

# Draw contours on the original image for visualization
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Save the results
cv2.imwrite('output_original_image_with_contours.png', image)
cv2.imwrite('output_grayscale_image.png', gray)
cv2.imwrite('output_binary_image.png', binary)

# Show the results
cv2.imshow('Original Image with Detected White Spots', image)
cv2.imshow('Grayscale Image', gray)
cv2.imshow('Binary Image', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
