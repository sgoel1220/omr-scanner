# import cv2
#
#
#
# image = cv2.imread("../omr.png")
#
# h, w, c = image.shape
#
# image = cv2.resize(image, (w//3, h//3))
# # cv2.imshow("output", image)
# # cv2.waitKey(0)
#
# orb = cv2.ORB_create(4000)
# key_point1, descriptors = orb.detectAndCompute(image, None)
#
#
#
# image_to_crop = cv2.imread("../omr_v3.jpeg")
# # cv2.imshow("result ",image_to_crop)
#
# key_point2, descriptors2 = orb.detectAndCompute(image_to_crop, None)
#
# image_key_point1 = cv2.drawKeypoints(image_to_crop, key_point2, None)
# #
# # cv2.imshow("result ",image_key_point1)
# # cv2.waitKey(0)
#
#
# bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# match = bf.match(descriptors2, descriptors)
#
# image_match  = cv2.drawMatches(image_to_crop, key_point2, image, key_point1, match, None, flags=2)
#
#
#
#
#


import cv2
import numpy as np
import os


def crop_matching_region(template_path, image_path, output_path, debug_dir="debug", threshold=8000):
    # Create debug directory if it doesn't exist
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # Load images
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if template is None or image is None:
        print("Error: Could not load images.")
        return

    # Save loaded images for debugging
    cv2.imwrite(os.path.join(debug_dir, "01_template.jpg"), template)
    cv2.imwrite(os.path.join(debug_dir, "02_image.jpg"), image)

    # Initialize ORB detector
    orb = cv2.ORB_create(threshold)

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(image, None)

    # Draw and save keypoints
    template_kp_img = cv2.drawKeypoints(template, kp1, None)
    image_kp_img = cv2.drawKeypoints(image, kp2, None)
    cv2.imwrite(os.path.join(debug_dir, "03_template_keypoints.jpg"), template_kp_img)
    cv2.imwrite(os.path.join(debug_dir, "04_image_keypoints.jpg"), image_kp_img)

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
    cv2.imwrite(os.path.join(debug_dir, "05_matches.jpg"), match_img)

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
    cropped = image[y:y + h, x:x + w]

    # Save the cropped image
    cv2.imwrite(output_path, cropped)
    print(f"Cropped region saved to {output_path}")


    # Save the bounding box for debugging
    img_with_bbox = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_with_bbox, [np.int32(dst)], True, (0, 255, 0), 3)
    cv2.imwrite(os.path.join(debug_dir, "06_bounding_box.jpg"), img_with_bbox)


# Example usage
# crop_matching_region("../input/omr.png", "../input/sample_omr_v4.jpeg", "output.jpg")
# crop_matching_region("../cropped_omr_v2.png", "output.jpg", "output_questions.jpg")
#
# for i in range(20):
#     crop_matching_region("../input/cropped_omr_v2.png", "output.jpg", f"output_questions_{i}.jpg", threshold= i * 1000)

# for i in range(20):
#     crop_matching_region("../input/omr.png", "../input/omr_v5.jpeg",  f"output_questions_{i}.jpg", threshold= i * 1000)

# crop_matching_region("../input/omr_cropped_v2.png", "output.jpg", f"output_questions_{0}.jpg", threshold= 3000)


# first user
# crop_matching_region("../input/omr.png", "../input/omr_v5.jpeg", "output.jpg", threshold=8000)
# crop_matching_region("../input/cropped_omr_v2.png", "output.jpg", "output_question.jpg", threshold=9000)
# for i in range(20):
#     crop_matching_region("../input/only_nums2.jpg", "../input/omr_v5.jpeg", f"output_questions_section_{i}.jpg",
#                          threshold= i * 1000)
