import cv2
import numpy as np

# Load the image
image_path = "question_bubbles.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
# blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles using HoughCircles
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=10,
    param1=50,
    param2=22,
    minRadius=5,
    maxRadius=10
)

# Draw detected circles
if circles is not None:
    print(circles)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw the circle
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)  # Draw the center

# Save the result
output_path = "detected_circles.png"
cv2.imwrite(output_path, image)

print(f"Processed image saved at: {output_path}")
