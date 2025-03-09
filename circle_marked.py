import numpy as np
import cv2
def transform_to_option_arr(section, image, output_path="./debug/041_lettered.png"):
    ans_arr = []
    for _ in range(len(section[0][0])):
        ans_arr.append([])

    for section in section:
        for options in section:
            for i, option in enumerate(options):
                ans_arr[i].append(option)
                cv2.putText(image, str(i), (option[0], option[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, image)
    return ans_arr



def detect_marked_circles(circles, image, current_running_option="a",
                          standard_deviation=2, output_path="./debug/05_marked_answers.jpg"):
    deviations = []
    intensities = []

    # Convert image to grayscale if not already
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Extract median intensity for each circle
    for (x, y, r) in circles:
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)

        # Get pixel intensities inside the circle
        pixels = gray[mask == 255]
        median_intensity = np.median(pixels)
        intensities.append(median_intensity)

    # Calculate the overall median intensity and standard deviation
    overall_median = np.median(intensities)
    std_dev = np.std(intensities)
    threshold = standard_deviation * std_dev  # Using 2 standard deviations

    # Identify marked circles
    marked_circles = []
    question_ans = {}
    for i, (x, y, r) in enumerate(circles):
        deviation = abs(intensities[i] - overall_median)
        deviations.append(deviation)

        if deviation > threshold:  # Threshold based on standard deviation
            question_ans[i] = current_running_option
            marked_circles.append((x, y, r))
            cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)  # Draw in black

            cv2.putText(image, current_running_option, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)


    # print("marked", marked_circles)
    # print("marked", len(marked_circles))
    # Save output image
    cv2.imwrite(output_path, image)

    return marked_circles, deviations, question_ans

def get_marked_options(circles, image, output_image_path="./debug/042_option_marked.png"):
    ans = ["" for _ in range(sum([len(i) + 10 for i in circles]))] # index is question id, value is option selected

    for i in range(len(circles)):
        # TODO edge case of 2 circles marked, pick the most deviation one
        _, _, cur_ans = detect_marked_circles(circles[i], image, current_running_option=chr(ord("a") + i),
                                              standard_deviation=1.2)

        for key in cur_ans:
            ans[key + 1] = cur_ans[key]


    return ans
    # detect_marked_circles(circles[0], image)


