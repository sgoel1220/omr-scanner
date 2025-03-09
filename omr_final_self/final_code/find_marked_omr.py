from typing import List

import cv2
import pprint
from final_code.add_missing_data import add_missing_circles
from final_code.circle_marked import transform_to_option_arr, get_marked_options
from final_code.circle_omr_options import cluster_circles
from final_code.final_score_calculator import cal_score, get_question_ans_mapping
from final_code.similarity_resize import crop_matching_region
# template_for_question = cv2.imread(r"C:\Users\shubh\Desktop\startup_ideas\omr_final_self\final_code\templates\template_for_questions.jpeg")
# template_for_omr = cv2.imread(r"C:\Users\shubh\Desktop\startup_ideas\omr_final_self\final_code\templates\template_for_whole_omr.jpeg")



def find_score_for_imr(input_file_path, template_for_question, template_for_omr, question_paper_id="V4073"):
    input_image = cv2.imread(input_file_path)

    cropped_with_omr = crop_matching_region(input_image, template_for_omr, "./debug/01_crop_to_omr.png", threshold=10000)
    cropped_with_question = crop_matching_region(cropped_with_omr, template_for_question,
                                                 "./debug/02_crop_to_questions.png", threshold=20000)

    section_circles, median_radius = cluster_circles(cropped_with_question.copy(), "./debug/03_cluster_circles.png")

    final_section_circles = add_missing_circles(cropped_with_question.copy(), section_circles, median_radius)
    option_marked_arr = transform_to_option_arr(final_section_circles, cropped_with_question.copy())

    marked_options = get_marked_options(option_marked_arr, cropped_with_question.copy())

    final_score, ans_matching = cal_score(marked_options, get_question_ans_mapping(question_paper_id))
    print(final_score)

    return final_score, ans_matching





# main("user_input/input_2.png")
