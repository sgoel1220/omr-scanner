import collections

import cv2
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cdist
# Load the image
# image_path = "question_bubbles.jpg"
image_path = "./final_code/debug/crop_to_questions.png"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)


def cluster_circles(image):

    # Convert to grayscale
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
            print(filtered_circles[i], filtered_labels[i])
            label_map[filtered_labels[i]].append(filtered_circles[i])
            cluster_idx = filtered_labels[i]
            cv2.circle(image, (x, y), r, colors[cluster_idx], 2)  # Circle outline
            cv2.circle(image, (x, y), 2, colors[cluster_idx], 3)  # Center point


        print(dict(label_map))

        # Save final labeled output
        output_path = "cluster.png"
        cv2.imwrite(output_path, image)
        print(f"Processed image saved at: {output_path}")

        return label_map

    return {}





cluster_map = cluster_circles(image)

circles = cluster_map[0]



# Step 1: Sort circles by y-coordinate
sorted_circles = sorted(circles, key=lambda c: c[1])

# Step 2: Group circles into rows based on median y-values
rows = []
current_row = [sorted_circles[0]]

for circle in sorted_circles[1:]:
    current_median_y = np.median([c[1] for c in current_row])  # Median y-value of the row
    if abs(current_median_y - circle[1]) < 10:  # Adjust threshold if needed
        current_row.append(circle)
    else:
        rows.append(current_row)
        current_row = [circle]

if current_row:
    rows.append(current_row)  # Add last row

# Step 3: Sort each row by x-coordinate
for row in rows:
    row.sort(key=lambda c: c[0])

# Step 4: Assign labels (A, B, C, D)
labels = ['A', 'B', 'C', 'D']
labeled_circles = []
for row in rows:
    for i, circle in enumerate(row):
        labeled_circles.append((*circle, labels[i]))

    # c += 1

image_path = "question_bubbles.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
# Step 6: Draw circles and labels
for x, y, r, label in labeled_circles:
    cv2.circle(image, (x, y), r, (255, 0, 0), 2)  # Blue circle
    cv2.putText(image, label, (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red label

# Step 7: Save the image
cv2.imwrite("labeled_circles.png", image)
print("Image saved as 'labeled_circles.png'")

# Step 8: Output labeled array
labeled_array = np.array(labeled_circles, dtype=object)
print(labeled_array)


# Step 1: Sort circles into rows based on their y-coordinate
# tolerance = 3  # Tolerance to consider y-coordinates as belonging to the same row
# rows = []
# # rows.append(circles[0])
# for circle in sorted(circles, key=lambda c: c[1]):  # Sort by y
#     added = False
#     if len(rows) == 0:
#         rows.append([circle])
#         continue
#     for row in rows:
#         if abs(row[0][1] - circle[1]) < tolerance:
#             row.append(circle)
#             added = True
#             break
#     # if not added:
#     #     rows.append([circle])
#
# # Step 2: Sort each row by x-coordinate
# for row in rows:
#     row.sort(key=lambda c: c[0])
#
# # Step 3: Assign labels (A, B, C, D)
# labels = ['A', 'B', 'C', 'D']
# labeled_circles = []
# c = 0
# for row in rows:
#     for i, circle in enumerate(row):
#         labeled_circles.append((*circle, str(c) +  labels[i]))
#
#     c += 1
#
# image_path = "question_bubbles.jpg"
# image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#
# # Step 5: Draw circles and labels
# for x, y, r, label in labeled_circles:
#     # cv2.circle(image, (x, y), r, (255, 0, 0), 2)  # Draw circle in blue
#     cv2.putText(image, label, (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red label
#
# # Step 6: Save the image
# cv2.imwrite("labeled_circles.png", image)
# print("Image saved as 'labeled_circles.png'")

# Step 7: Output labeled array
# labeled_array = np.array(labeled_circles, dtype=object)
# print(labeled_array)