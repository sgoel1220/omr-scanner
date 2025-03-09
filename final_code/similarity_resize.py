import cv2
import os
import numpy as np

def crop_matching_region(og_image, og_template, output_path, debug_dir="debug", threshold=8000):
    # Create debug directory if it doesn't exist
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    if og_template is None or og_image is None:
        print("Error: Could not load images.")
        return
    # Load images
    # template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(og_template, cv2.COLOR_BGR2GRAY)
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)



    # Save loaded images for debugging
    # cv2.imwrite(os.path.join(debug_dir, "01_template.jpg"), template)
    # cv2.imwrite(os.path.join(debug_dir, "02_image.jpg"), image)

    # Initialize ORB detector
    orb = cv2.ORB_create(threshold)

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(image, None)

    # Draw and save keypoints
    template_kp_img = cv2.drawKeypoints(template, kp1, None)
    image_kp_img = cv2.drawKeypoints(image, kp2, None)
    # cv2.imwrite(os.path.join(debug_dir, "03_template_keypoints.jpg"), template_kp_img)
    # cv2.imwrite(os.path.join(debug_dir, "04_image_keypoints.jpg"), image_kp_img)

    # Use BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        print("Not enough matches found.")
        return

    # Draw and save matches
    match_img = cv2.drawMatches(template, kp1, image, kp2, matches[:20], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imwrite(os.path.join(debug_dir, "05_matches.jpg"), match_img)

    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get dimensions of the template
    h, w = template.shape

    # Define the corners of the template in the original image space
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

    # Get bounding box for cropping
    x, y, w, h = cv2.boundingRect(dst)

    # Crop the matching region
    cropped = og_image[y:y + h, x:x + w]

    # Save the cropped image
    cv2.imwrite(output_path, cropped)
    print(f"Cropped region saved to {output_path}")


    # Save the bounding box for debugging
    img_with_bbox = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_with_bbox, [np.int32(dst)], True, (0, 255, 0), 3)
    # cv2.imwrite(os.path.join(debug_dir, "bounding_box.jpg"), img_with_bbox)


    return cropped