from evaluation.metrics import classification_report, confusion_matrix
from const import ConceptType, type_mapping


def classification_summary(y_true, y_pred):
    print(y_true, y_pred)
    try:
        cls_rep = classification_report(y_true, y_pred)
        print(cls_rep)
    except ValueError:
        print("Value error!")
        return None
    print(confusion_matrix(y_true, y_pred, [type_mapping[ConceptType.ACTOR_NAME.value],
                                            type_mapping[ConceptType.ACTOR_INSTANCE.value],
                                            type_mapping[ConceptType.ACTION_NAME.value],
                                            type_mapping[ConceptType.ACTION_INSTANCE.value],
                                            type_mapping[ConceptType.ACTION_STATUS.value],
                                            type_mapping[ConceptType.BO_NAME.value],
                                            type_mapping[ConceptType.BO_INSTANCE.value],
                                            type_mapping[ConceptType.BO_STATUS.value],
                                            ConceptType.OTHER.value]))
    return classification_report(y_true, y_pred, output_dict=True)


def print_attribute_results(results, ground_truths,):
    avg_rep = []
    print(results.keys())
    for key in results.keys():
        trues = []
        preds = []
        curr_res = results[key]
        for att in ground_truths[key].keys():
            if att in curr_res[4].keys():
                trues.append(ground_truths[key][att])
                preds.append(curr_res[4][att])
            print("for log " + key)
            avg_rep.append(classification_summary(trues, preds))