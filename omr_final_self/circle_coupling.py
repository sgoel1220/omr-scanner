import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Load the image
image_path = "question_bubbles.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect circles using HoughCircles
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=10,
    param1=50,
    param2=20,
    minRadius=5,
    maxRadius=10
)

# Process detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    circle_centers = circles[0, :, :2]  # Extract (x, y) coordinates

    # Use only the X-coordinates for clustering into 4 vertical sections
    x_values = circle_centers[:, 0].reshape(-1, 1)

    # Apply K-Means clustering with 4 vertical groups
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(x_values)
    cluster_centers = kmeans.cluster_centers_

    # Calculate distances of each point to its cluster center
    distances = cdist(x_values, cluster_centers)

    # Find the minimum distance for each point (i.e., its cluster assignment)
    min_distances = np.min(distances, axis=1)

    # Use the Interquartile Range (IQR) method to filter out outliers
    Q1 = np.percentile(min_distances, 25)
    Q3 = np.percentile(min_distances, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR  # Remove points that are too far (increase 1.5 to ignore more)

    # Keep only the circles that are within the threshold distance
    filtered_indices = np.where(min_distances <= threshold)[0]
    filtered_circles = circles[0][filtered_indices]
    filtered_labels = labels[filtered_indices]

    # Assign colors to different clusters
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow

    # Draw only filtered circles
    for i, (x, y, r) in enumerate(filtered_circles):
        cluster_idx = filtered_labels[i]
        cv2.circle(image, (x, y), r, colors[cluster_idx], 2)  # Circle outline
        cv2.circle(image, (x, y), 2, colors[cluster_idx], 3)  # Center point

    # Save the result
    output_path = "filtered_vertical_sections.png"
    cv2.imwrite(output_path, image)
    print(f"Processed image saved at: {output_path}")
    print(f"Original Circles: {len(circles[0])}, Filtered Circles: {len(filtered_circles)}")

else:
    print("No circles detected.")
