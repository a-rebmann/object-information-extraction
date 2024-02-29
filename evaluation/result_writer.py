from const import AttributeType
import time
import csv


def write_bert_results(OUT_DIR, CHUNK_WISE, allowed_tags, results):
    results_file = OUT_DIR + "bert_results_" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        if CHUNK_WISE:
            header = ['.model', 'log'] + \
                     ['f1-score ' + tag for tag in allowed_tags] + \
                     ['support ' + tag for tag in allowed_tags] + \
                     ['weighted support ' + tag for tag in allowed_tags]
            writer.writerow(header)
        else:
            header = ['.model', 'log', 'entity-level-f1', 'entity-level-prec', 'entity-level-rec', 'conf'] + \
                     ['f1-score ' + tag for tag in allowed_tags] + \
                     ['precision ' + tag for tag in allowed_tags] + \
                     ['recall ' + tag for tag in allowed_tags] + \
                     ['support ' + tag for tag in allowed_tags] + \
                     ['weighted support ' + tag for tag in allowed_tags]
            writer.writerow(header)
        for key in results.keys():
            if CHUNK_WISE:
                writer.writerow(['bert', key] +
                                [results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['support'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['support'] *
                                 results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags])
            else:
                writer.writerow(['bert', key, results[key].bert_entity_level_f1, results[key].bert_entity_level_prec,
                                 results[key].bert_entity_level_rec, results[key].conf_matrix] +
                                [results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['precision'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['recall'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['support'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['support'] *
                                 results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags])



def write_baseline_results(overall_result, results, OUT_DIR, CHUNK_WISE, allowed_tags):
    results_file = OUT_DIR + "baseline_results_" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        if CHUNK_WISE:
            header = ['.model', 'log'] + \
                     ['f1-score ' + tag for tag in allowed_tags] + \
                     ['precision ' + tag for tag in allowed_tags] + \
                     ['recall ' + tag for tag in allowed_tags] + \
                     ['support ' + tag for tag in allowed_tags]
            writer.writerow(header)
        else:
            header = ['.model', 'log', 'entity-level-f1', 'entity-level-prec', 'entity-level-rec', 'conf'] + \
                     ['f1-score ' + tag for tag in allowed_tags] + \
                     ['precision ' + tag for tag in allowed_tags] + \
                     ['recall ' + tag for tag in allowed_tags] + \
                     ['support ' + tag for tag in allowed_tags]
            writer.writerow(header)
        if CHUNK_WISE:
            writer.writerow(['hmm', 'overall'] + \
                            [overall_result['HMM'].bert_per_class_metrics[tag][
                                 'f1-score'] if tag in overall_result['HMM'].available_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result['HMM'].bert_per_class_metrics[tag][
                                 'precision'] if tag in overall_result['HMM'].available_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result['HMM'].bert_per_class_metrics[tag][
                                 'recall'] if tag in overall_result['HMM'].available_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result['HMM'].bert_per_class_metrics[tag][
                                 'support'] if tag in overall_result['HMM'].available_tags else 0 for tag in
                             allowed_tags])

            writer.writerow(['bert', 'overall'] + \
                            [overall_result['BERT'].bert_per_class_metrics[tag][
                                 'f1-score'] if tag in overall_result['BERT'].available_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result['BERT'].bert_per_class_metrics[tag][
                                 'precision'] if tag in overall_result['BERT'].available_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result['BERT'].bert_per_class_metrics[tag][
                                 'recall'] if tag in overall_result['BERT'].available_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result['BERT'].bert_per_class_metrics[tag][
                                 'support'] if tag in overall_result['BERT'].available_tags else 0 for tag in
                             allowed_tags])
        else:
            writer.writerow(['hmm', 'overall', overall_result.hmm_entity_level_f1, overall_result.hmm_entity_level_prec,
                             overall_result.hmm_entity_level_rec, overall_result.hmm_conf_matrix] + \
                            [overall_result.hmm_per_class_metrics[tag][
                                 'f1-score'] if tag in overall_result.available_hmm_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result.hmm_per_class_metrics[tag][
                                 'precision'] if tag in overall_result.available_hmm_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result.hmm_per_class_metrics[tag][
                                 'recall'] if tag in overall_result.available_hmm_tags else 0 for tag in allowed_tags] + \
                            [overall_result.hmm_per_class_metrics[tag][
                                 'support'] if tag in overall_result.available_hmm_tags else 0 for tag in allowed_tags])
            writer.writerow(
                ['bert', 'overall', overall_result.bert_entity_level_f1, overall_result.bert_entity_level_prec,
                 overall_result.bert_entity_level_rec, overall_result.bert_conf_matrix] + \
                [overall_result.bert_per_class_metrics[tag][
                     'f1-score'] if tag in overall_result.available_bert_tags else 0 for tag in allowed_tags] + \
                [overall_result.bert_per_class_metrics[tag][
                     'precision'] if tag in overall_result.available_bert_tags else 0 for tag in allowed_tags] + \
                [overall_result.bert_per_class_metrics[tag][
                     'recall'] if tag in overall_result.available_bert_tags else 0 for tag in allowed_tags] + \
                [overall_result.bert_per_class_metrics[tag][
                     'support'] if tag in overall_result.available_bert_tags else 0 for tag in allowed_tags])
        for key in results.keys():
            if CHUNK_WISE:
                if 'HMM' in key:
                    writer.writerow(['hmm', key] +
                                    [results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags] +
                                    [results[key].bert_per_class_metrics[tag]['precision'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags] +
                                    [results[key].bert_per_class_metrics[tag]['recall'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags] +
                                    [results[key].bert_per_class_metrics[tag]['support'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags])
                else:
                    writer.writerow(['bert', key] +
                                    [results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags] +
                                    [results[key].bert_per_class_metrics[tag]['precision'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags] +
                                    [results[key].bert_per_class_metrics[tag]['recall'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags] +
                                    [results[key].bert_per_class_metrics[tag]['support'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags])
            else:
                writer.writerow(['hmm', key, results[key].hmm_entity_level_f1, results[key].hmm_entity_level_prec,
                                 results[key].hmm_entity_level_rec, results[key].hmm_conf_matrix] +
                                [results[key].hmm_per_class_metrics[tag]['f1-score'] if tag in results[
                                    key].available_hmm_tags else 0 for tag in allowed_tags] +
                                [results[key].hmm_per_class_metrics[tag]['precision'] if tag in results[
                                    key].available_hmm_tags else 0 for tag in allowed_tags] +
                                [results[key].hmm_per_class_metrics[tag]['recall'] if tag in results[
                                    key].available_hmm_tags else 0 for tag in allowed_tags] +
                                [results[key].hmm_per_class_metrics[tag]['support'] if tag in results[
                                    key].available_hmm_tags else 0 for tag in allowed_tags])
                writer.writerow(['bert', key, results[key].bert_entity_level_f1, results[key].bert_entity_level_prec,
                                 results[key].bert_entity_level_rec, results[key].bert_conf_matrix] +
                                [results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                    key].available_bert_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['precision'] if tag in results[
                                    key].available_bert_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['recall'] if tag in results[
                                    key].available_bert_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['support'] if tag in results[
                                    key].available_bert_tags else 0 for tag in allowed_tags])


def write_gold(gt, OUT_DIR):
    results_file = OUT_DIR + "gold_" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        header = ['log', 'type', 'true']
        writer.writerow(header)
        for key in gt.keys():
            for att in gt[key].keys():
                writer.writerow([key, att, gt[key][att]])


def write_attribute_annotations(att_annots, OUT_DIR):
    annots_file = OUT_DIR + "attribute_annotation_results_" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(annots_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        header = ['log', 'attributeclassification', 'annotation']
        writer.writerow(header)
        for key in att_annots.keys():
            for att in att_annots[key].keys():
                writer.writerow([key, att, att_annots[key][att]])


def write_attribute_results(results, ground_truths, att_annots, OUT_DIR, with_samples=False):
    results_file = OUT_DIR + "attribute_classification_results_" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    avg_rep = []
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        header = ['log', 'attribute', 'type', 'true', 'pred', 'approach', 'score']
        writer.writerow(header)
        print(results.keys())
        for key in results.keys():
            trues = []
            preds = []
            curr_res = results[key]
            for att in ground_truths[key].keys():
                if att in curr_res[1].keys():
                    writer.writerow([key, att, att_annots[key][att] if att in att_annots[
                        key].keys() else AttributeType.UNKNOWN.value, ground_truths[key][att], curr_res[1][att][0],
                                     'semantic similarity', curr_res[1][att][1]])
                if att in curr_res[3].keys():
                    writer.writerow([key, att, att_annots[key][att] if att in att_annots[
                        key].keys() else AttributeType.UNKNOWN.value, ground_truths[key][att], curr_res[3][att],
                                     'bert combined', 'nan'])
                if att in curr_res[4].keys():
                    trues.append(ground_truths[key][att])
                    preds.append(curr_res[4][att])
                    writer.writerow([key, att, att_annots[key][att] if att in att_annots[
                        key].keys() else AttributeType.UNKNOWN.value, ground_truths[key][att], curr_res[4][att],
                                     'final', 'nan'])
                if att in curr_res[5].keys() and with_samples:
                    writer.writerow([key, att, att_annots[key][att] if att in att_annots[
                        key].keys() else AttributeType.UNKNOWN.value, ground_truths[key][att], curr_res[5][att],
                                     'samples', 'nan'])