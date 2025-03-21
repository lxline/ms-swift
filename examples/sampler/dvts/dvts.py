def collect_sft_data(previous_texts, previous_scores, root):
    answers = []
    if root['current_text'] == "":
        current_texts = previous_texts[:]
        current_scores = previous_scores[:]
    else:
        current_texts = previous_texts[:] + [root['current_text']]
        current_scores = previous_scores[:] + [root['current_score']]
    if len(root['children']) == 0:
        if root['terminated'] and root["outcome_score"] > 0:
            answers.append((current_texts, current_scores))
    else:
        if len(current_texts) == 0 or any([child['terminated'] for child in root['children']]):
            for child in root['children']:
                answers += collect_sft_data(current_texts, current_scores, child)
        else:
            best = max(root['children'], key=lambda x: x['current_score'] + x['outcome_score'])
            if best['current_score'] > 0.5 or best['outcome_score'] > 0:
                answers += collect_sft_data(current_texts, current_scores, best)
    return answers


class dpo_data:
    def __init__(self,
                 previous_texts,
                 previous_scores,
                 chosen_text,
                 chosen_score,
                 rejected_text,
                 rejected_score):
        self.previous_texts = previous_texts
        self.previous_scores = previous_scores
        self.chosen_text = chosen_text
        self.chosen_score = chosen_score
        self.rejected_text = rejected_text
        self.rejected_score = rejected_score


def collect_dpo_data(previous_texts, previous_scores, root):
    answers = []
    if root['current_text'] == "":
        current_texts = previous_texts[:]
        current_scores = previous_scores[:]
    else:
        current_texts = previous_texts[:] + [root['current_text']]
        current_scores = previous_scores[:] + [root['current_score']]

    if len(root['children']) > 0:
        correct = False
        sorted_children = sorted(root['children'], key=lambda x: x['current_score'] + x['outcome_score'])
        best_index, worst_index = 0, 0
        for index in range(1, len(root['children'])):
            if root['children'][index]['current_score'] > root['children'][best_index]['current_score']:
                best_index = index
            if root['children'][index]['current_score'] < root['children'][worst_index]['current_score']:
                worst_index = index
        for index in range(len(root['children'])):
            child = root['children'][index]
            if child['current_score'] > 0.5:
                (child_answers, child_correct) = collect_dpo_data(current_texts, current_scores, child)
                if child_correct:
                    answers += child_answers[:]
                    correct = True
                    if (index == best_index
                            and root['children'][best_index]['current_score'] - root['children'][worst_index][
                                'current_score'] > 0.3):
                        data = dpo_data(current_texts,
                                        current_scores,
                                        root['children'][best_index]['current_text'],
                                        root['children'][best_index]['current_score'],
                                        root['children'][worst_index]['current_text'],
                                        root['children'][worst_index]['current_score'])
                        answers.append(data)
    else:
        correct = root["outcome_score"] > 0
    return answers, correct