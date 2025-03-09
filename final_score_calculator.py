import collections

import requests
import requests

from constants import ANSWER_BANK_URL

def get_question_ans_mapping(question_paper_id):
    """Fetch data from a specific Google Sheet."""
    try:
        url  =f"{ANSWER_BANK_URL}?sheet={question_paper_id}"
        print(url)
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        data = response.json()  # Convert JSON response to Python dictionary
        if 'error' in data:
            raise RuntimeError(f"Answer bank for question-id {question_paper_id} not found")
        question_to_ans_mapping = collections.defaultdict(lambda: "")
        for cur_d in data:
            question_to_ans_mapping[cur_d['question']] = cur_d['answer']
        return question_to_ans_mapping
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def cal_score(marked_options, ans_key, right_mark=1, wrong_mark=0):
    total = 0
    marked_options = [""] + marked_options
    ans_matching = {} #question: (was_correct, blank, was_wrong), marked ans, actual ans
    for i in range(1, len(marked_options)):
        if not i in ans_key:
            continue
        if ans_key[i] and ans_key[i].lower() == marked_options[i].lower():
            total += right_mark
            ans_matching[i] = ("CORRECT", marked_options[i].lower(), ans_key[i])
        else:
            if marked_options[i] == "":
                ans_matching[i] = ("BLANK", marked_options[i].lower(), ans_key[i])
                continue
            ans_matching[i] = ("WRONG", marked_options[i].lower(), ans_key[i])
            total += wrong_mark

        print(ans_matching, marked_options[i])

    return total, ans_matching