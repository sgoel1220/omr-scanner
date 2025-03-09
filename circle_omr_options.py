import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import collections


def cluster_circles(og_image, output_path="cluster_circle.png"):
    # Detect circles using HoughCircles
    image = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=50,
        param2=18,
        minRadius=5,
        maxRadius=10
    )

    if circles is None:
        raise RuntimeError("No circles found")
    # Process detected circles
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
    threshold = Q3 + 0.6 * IQR  # Remove points that are too far (increase 1.5 to ignore more)

    # Keep only the circles that are within the threshold distance
    filtered_indices = np.where(min_distances <= threshold)[0]
    filtered_circles = circles[0][filtered_indices]
    filtered_labels = labels[filtered_indices]

    # Assign colors to different clusters
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow

    label_map = collections.defaultdict(list)

    # Draw only filtered circles
    for i, (x, y, r) in enumerate(filtered_circles):
        # print(filtered_circles[i], filtered_labels[i])
        label_map[filtered_labels[i]].append(filtered_circles[i])
        cluster_idx = filtered_labels[i]
        cv2.circle(og_image, (x, y), r, colors[cluster_idx], 2)  # Circle outline
        cv2.circle(og_image, (x, y), 2, colors[cluster_idx], 3)  # Center point


    cv2.imwrite(output_path, og_image)
    # print(f"Processed image saved at: {output_path}")
    median_radius = int(np.median(circles[0, :, 2]))  # Ensure integer

    return label_map, median_radius



