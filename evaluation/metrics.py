import itertools
from sklearn.metrics import f1_score as f1_score_sklearn
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import classification_report, confusion_matrix


def f1_entity_level(true, pred):
    """
    :param true: the true entities as list of lists [[...],[...],...]
    :param pred: the predicted entities as list of lists [[...],[...],...]
    :return: the f1 score on entity level
    """
    return f1_score(true, pred)


def prec_entity_level(true, pred):
    """
    :param true: the true entities as list of lists [[...],[...],...]
    :param pred: the predicted entities as list of lists [[...],[...],...]
    :return: the precision score on entity level
    """
    return precision_score(true, pred)


def rec_entity_level(true, pred):
    """
    :param true: the true entities as list of lists [[...],[...],...]
    :param pred: the predicted entities as list of lists [[...],[...],...]
    :return: the recall score on entity level
    """
    return recall_score(true, pred)


def f1_token_level(true_labels, predictions):
    true_labels = list(itertools.chain(*true_labels))
    predictions = list(itertools.chain(*predictions))

    labels = list(set(true_labels) - {'[PAD]', 'O'})

    return f1_score_sklearn(true_labels,
                            predictions,
                            average='micro',
                            labels=labels)


def per_class_metrics(true_labels, predictions):
    true_labels = list(itertools.chain(*true_labels))
    predictions = list(itertools.chain(*predictions))
    cnt = {}
    unique_labels = []
    for label in true_labels:
        if label in cnt:
            cnt[label] += 1
        else:
            unique_labels.append(label)
            cnt[label] = 1
    for label in predictions:
        if label in unique_labels:
            continue
        else:
            unique_labels.append(label)
    clz_rpt = classification_report(true_labels, predictions, output_dict=True)
    return clz_rpt, unique_labels


def conf_matrix(true_labels, predictions, labels):
    #true_labels = list(itertools.chain(*true_labels))
    #predictions = list(itertools.chain(*predictions))
    return confusion_matrix(true_labels, predictions, labels=labels)
