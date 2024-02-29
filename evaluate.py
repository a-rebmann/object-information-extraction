import csv
import time
import sys
import os
from copy import deepcopy
from datetime import datetime
from statistics import mean

import ocel
from nltk import WordNetLemmatizer

from config import Config
import pandas as pd
from instancediscovery.instancediscoverer import InstanceDiscoverer
from const import *
from evaluation.flatten_ocel import flatten_log
from evaluation.flattened_log_gs_util import mapping
from evaluation.eval_result import EvaluationResult
from preprocessing.preprocessor import preprocess_label
from propertytoinstance.propertyassignment import PropertyAssigner
from readwrite import loader, writer
from main import DEFAULT_RES_DIR, DEFAULT_MODEL_DIR, augment_and_transform_log

logs = [
    "BPI_Challenge_2019.xes",
    "BPI_Challenge_2017.xes",
    "runningexample.jsonocel"
]

model_idx = {
    "BPI_Challenge_2019.xes": 3,
    "BPI_Challenge_2017.xes": 4,
    "runningexample.jsonocel": 16
}

obj_type_mapping = {
    "runningexample.jsonocel": {"orders": "order", "items": "item", "packages": "package", "products": "product",
                                "customers": "customer"}
}

ocel_logs = ["runningexample.jsonocel"]

EVAL_INPUT_DIR = 'input/evaluation/'

BERT_DIR_ON_MACHINE = '/Users/adrianrebmann/Develop/semantic_event_parsing/fine-tuning-bert-for-semantic-labeling/model/'

EVAL_OUTPUT_DIR = 'output/evaluation/'

DEFAULT_CONFIG = Config(input_dir=EVAL_INPUT_DIR, model_dir=BERT_DIR_ON_MACHINE, resource_dir=DEFAULT_RES_DIR,
                        output_dir=EVAL_OUTPUT_DIR, bo_ratio=0.5, conf_thresh=0.5, instance_thresh=0.5,
                        exclude_data_origin=[], we_file="glove-wiki-gigaword-50", matching_mode=MatchingMode.SEM_SIM,
                        mode=EVAL_MODE, res_type=False)

confs = [
    DEFAULT_CONFIG
]


def set_current_run(filename):
    current_run = filename
    if "runningexample" in current_run:
        current_run = "runningexample.jsonocel"
    return current_run


def prepare_logs_for_rediscovery():
    import json
    lemmatizer = WordNetLemmatizer()
    list_of_files = {}
    for (dir_path, dir_names, filenames) in os.walk(DEFAULT_CONFIG.input_dir + "raw/"):
        for filename in filenames:
            if ".jsonocel" in filename:
                ocel_log = loader.load_ocel(dir_path, filename)
                for obj in list(ocel.get_object_types(ocel_log)):
                    print(filename, obj)
                    flat_name, covered_object_types = flatten_log(ocel_log, dir_path, filename, obj)
                    print(covered_object_types)
                    list_of_files[flat_name] = os.sep.join([dir_path])
                    gold_standard_template = {}
                    attribute_name_to_true_label = {}
                    attribute_name_to_object = {}
                    attributes = deepcopy(ocel.get_attribute_names(ocel_log))
                    for oid, objty in ocel.get_objects(ocel_log).items():
                        if objty[OCEL_TYPE] not in covered_object_types:
                            continue
                        for prop in objty[OCEL_OVMAP].keys():
                            if prop in attributes:
                                attribute_name_to_true_label[prop] = OBJ_PROP
                                if prop not in attribute_name_to_object:
                                    attribute_name_to_object[prop] = [objty[OCEL_TYPE]]
                                elif objty[OBJ_TYPE] not in attribute_name_to_object[prop]:
                                    attribute_name_to_object[prop].append(objty[OCEL_TYPE])
                                attributes.remove(prop)
                    for att in attributes:
                        attribute_name_to_true_label[att] = EVENT_ATT
                        attribute_name_to_object[att] = []
                    if filename in mapping.keys():
                        gold_standard_template[OBJ_TYPE] = [
                            (mapping[filename][ty] if ty in mapping[filename].keys() else ty) for ty in
                            covered_object_types]
                    else:
                        gold_standard_template[OBJ_TYPE] = [lemmatizer.lemmatize(ty) for ty in covered_object_types]

                    attribute_name_to_true_label[XES_CASE] = OBJ_INST
                    attribute_name_to_object[XES_CASE] = [lemmatizer.lemmatize(obj)]
                    for inst_att in covered_object_types:
                        if inst_att != obj:
                            attribute_name_to_object[inst_att] = [lemmatizer.lemmatize(inst_att)]
                            attribute_name_to_true_label[inst_att] = OBJ_INST

                    gold_standard_template["attibutes_to_category"] = attribute_name_to_true_label
                    gold_standard_template["attibutes_to_obj_type"] = attribute_name_to_object
                    # gold_standard_template["attibutes_to_obj_type"] = attribute_name_to_objecty

                    with open(DEFAULT_CONFIG.input_dir + "gold/" + flat_name + '_gs.json', 'w', encoding='utf-8') as f:
                        json.dump(gold_standard_template, f, ensure_ascii=False, indent=4)
            else:
                continue
    print(list_of_files)


def get_gold_standard(path):
    import json
    # Opening JSON file
    f = open(path)
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    # Closing file
    f.close()
    return data


def get_object_type_scores(eval_result, predictions, gold_standard, log):
    object_type_tp = sum([1 for obj in gold_standard[OBJ_TYPE] if obj in predictions[OBJ_TYPE]])
    object_type_fp = sum([1 for obj in predictions[OBJ_TYPE] if obj not in gold_standard[OBJ_TYPE]])
    object_type_fn = sum([1 for obj in gold_standard[OBJ_TYPE] if obj not in predictions[OBJ_TYPE]])
    obj_type_precision = 0
    obj_type_recall = 0
    obj_type_f1 = 0
    if object_type_tp > 0 or object_type_fp > 0:
        obj_type_precision = object_type_tp / (object_type_tp + object_type_fp)
    if object_type_tp > 0 or object_type_fn > 0:
        obj_type_recall = object_type_tp / (object_type_tp + object_type_fn)
    if obj_type_precision > 0 or obj_type_recall > 0:
        obj_type_f1 = (2 * obj_type_precision * obj_type_recall) / (obj_type_recall + obj_type_precision)
    print(gold_standard[OBJ_TYPE], "True")
    print(log.object_types, "Pred")

    print("Object type precision", obj_type_precision)
    print("Object type recall", obj_type_recall)
    print("Object type f1", obj_type_f1)
    print("*" * 40)
    eval_result.obj_type_precision = obj_type_precision
    eval_result.obj_type_recall = obj_type_recall
    eval_result.obj_type_f1 = obj_type_f1
    return obj_type_precision, obj_type_recall, obj_type_f1, len(gold_standard[OBJ_TYPE])


def get_object_instance_scores(eval_result, predictions, gold_standard, ocel_log, aug_log, log,
                               instance_discoverer: InstanceDiscoverer, obj_type_occurrences=None,
                               only_correct_types=True):
    num_instances_per_obj_true = {}
    print(obj_type_occurrences)
    if obj_type_occurrences:
        num_instances_per_obj_true = {obj_type: num for obj_type, num in obj_type_occurrences.items() if
                                      obj_type in gold_standard[OBJ_TYPE]}
        if only_correct_types:
            num_instances_per_obj_true = {obj_type: num for obj_type, num in num_instances_per_obj_true.items() if
                                          obj_type in log.object_types}
    else:
        for att, cat in gold_standard["attibutes_to_category"].items():
            if cat == OBJ_INST:
                for att_o, obj in gold_standard["attibutes_to_obj_type"].items():
                    if att == att_o:
                        if len(obj) == 0:
                            print(att_o)
                            continue
                        try:
                            unique_for_att = [val for val in aug_log.df[att].unique() if
                                              preprocess_label(val) not in TERMS_FOR_MISSING]
                            num_instances_per_obj_true[obj[0]] = len(unique_for_att)
                        except KeyError:
                            print("KeyError", att)
                            num_instances_per_obj_true[obj[0]] = 0
        for obj_type, num in gold_standard["implicit_instances"].items():
            if obj_type not in num_instances_per_obj_true:
                num_instances_per_obj_true[obj_type] = num

    num_instances_per_obj_pred = {bo: sum([1 for _, obj in ocel.get_objects(ocel_log).items() if obj[OCEL_TYPE] == bo])
                                  for bo in predictions[OBJ_TYPE]}
    if only_correct_types:
        num_instances_per_obj_pred = {obj_type: num for obj_type, num in num_instances_per_obj_pred.items() if
                                      obj_type in gold_standard[OBJ_TYPE]}
    print("INSTANCES")
    print(instance_discoverer.object_to_id)
    print(num_instances_per_obj_true, "True")
    print(num_instances_per_obj_pred, "Pred")
    print("-" * 40)

    count = sum([num for _, num in num_instances_per_obj_true.items()])
    # the number of true positive instances is the sum of the (minimum of number of discovered instances and actual
    # instances) per type that is actually in the data
    object_inst_tp = sum(
        [min(num, num_instances_per_obj_true[obj]) for obj, num in num_instances_per_obj_pred.items() if
         obj in num_instances_per_obj_true])
    # the number of false positive instances is the sum of the instances of object types that were falsely discovered
    # as such plus additional instances for object types that are in the data.
    object_inst_fp = sum(
        [num for obj, num in num_instances_per_obj_pred.items() if obj not in num_instances_per_obj_true]) + sum(
        [max(0, num - num_instances_per_obj_true[obj]) for obj, num in num_instances_per_obj_pred.items() if
         obj in num_instances_per_obj_true])
    # the number of false negative instances is the sum of instances of object types that weren't extracted plus
    # missing instances of object types that were discovered
    object_inst_fn = sum(
        [num for obj, num in num_instances_per_obj_true.items() if obj not in predictions[OBJ_TYPE]]) + sum(
        [max(0, num_instances_per_obj_true[obj] - num) for obj, num in num_instances_per_obj_pred.items() if
         obj in num_instances_per_obj_true])
    obj_inst_precision = 0
    obj_inst_recall = 0
    obj_inst_f1 = 0
    if object_inst_tp > 0 or object_inst_fp > 0:
        obj_inst_precision = object_inst_tp / (object_inst_tp + object_inst_fp)
    if object_inst_tp > 0 or object_inst_fn > 0:
        obj_inst_recall = object_inst_tp / (object_inst_tp + object_inst_fn)
    if obj_inst_precision > 0 or obj_inst_recall > 0:
        obj_inst_f1 = (2 * obj_inst_precision * obj_inst_recall) / (obj_inst_recall + obj_inst_precision)
    print("Object instance precision", obj_inst_precision)
    print("Object instance recall", obj_inst_recall)
    print("Object instance f1", obj_inst_f1)
    print("*" * 40)
    eval_result.obj_inst_precision = obj_inst_precision
    eval_result.obj_inst_recall = obj_inst_recall
    eval_result.obj_inst_f1 = obj_inst_f1
    return obj_inst_precision, obj_inst_recall, obj_inst_f1, count


def get_instance_prop_scores(eval_result, gold_standard, log, property_assigner: PropertyAssigner,
                             only_correct_types=True, config=None):
    print("PROPERTIES")
    print({k: v for k, v in gold_standard["attibutes_to_obj_type"].items() if
           gold_standard["attibutes_to_category"][k] == OBJ_PROP}, "True")
    print(property_assigner.object_to_property, "Pred")
    print("-" * 40)
    inst_prop_tp = 0

    inst_prop_fp = 0

    inst_prop_fn = 0

    count = 0

    for att, cat in gold_standard["attibutes_to_category"].items():
        if cat == OBJ_PROP:
            count += 1
            for att_o, obj in gold_standard["attibutes_to_obj_type"].items():
                if att == att_o:
                    for o_plain in obj:
                        o = property_assigner.lemmatizer.lemmatize(o_plain)
                        if (only_correct_types and o not in log.object_types) or (
                                config and o_plain in config.mask_attribute and o_plain not in ["items", "orders",
                                                                                                "packages"]):
                            continue
                        if o in property_assigner.object_to_property and att_o in property_assigner.object_to_property[
                            o]:
                            inst_prop_tp = inst_prop_tp + 1
                        elif o not in property_assigner.object_to_property.keys() or att_o not in \
                                property_assigner.object_to_property[o]:
                            inst_prop_fn = inst_prop_fn + 1
    for obj_type, atts in property_assigner.object_to_property.items():
        for att in atts:
            if att in gold_standard["attibutes_to_obj_type"] and obj_type not in gold_standard["attibutes_to_obj_type"][
                att]:
                inst_prop_fp = inst_prop_fp + 1

    inst_prop_precision = 0
    inst_prop_recall = 0
    inst_prop_f1 = 0
    if inst_prop_tp > 0 or inst_prop_fp > 0:
        inst_prop_precision = inst_prop_tp / (inst_prop_tp + inst_prop_fp)
    if inst_prop_tp > 0 or inst_prop_fn > 0:
        inst_prop_recall = inst_prop_tp / (inst_prop_tp + inst_prop_fn)
    if inst_prop_precision > 0 or inst_prop_recall > 0:
        inst_prop_f1 = (2 * inst_prop_precision * inst_prop_recall) / (inst_prop_recall + inst_prop_precision)
    print("Instance-to-property precision", inst_prop_precision)
    print("Instance-to-property recall", inst_prop_recall)
    print("Instance-to-property f1", inst_prop_f1)
    print("*" * 40)
    eval_result.obj_prop_precision = inst_prop_precision
    eval_result.obj_prop_recall = inst_prop_recall
    eval_result.obj_prop_f1 = inst_prop_f1
    return inst_prop_precision, inst_prop_recall, inst_prop_f1, count


def get_old_id_from_new(new_event, aug_log, goldstandard):
    old_ids = {}
    for new_id in new_event.omap:
        if isinstance(new_id, str) and len(new_id.split("#")) == 3:
            # print(new_id.split("#")[1])
            # print(goldstandard["attibutes_to_obj_type"])
            # print(datetime.fromisoformat(new_id.split("#")[0]), new_id)
            id_attribute = [k for k, v in goldstandard["attibutes_to_obj_type"].items() if
                            new_id.split("#")[1] in v and goldstandard["attibutes_to_category"][k] == OBJ_INST]

            if len(id_attribute) > 0:
                id_attribute = id_attribute[0]
                if new_id.split("#")[1] not in old_ids:
                    old_ids[new_id] = set()  # new_id.split("#")[1]
                # print(datetime.fromisoformat(new_id.split("#")[0]))
                filtered = aug_log.df.loc[
                    (aug_log.df[aug_log.timestamp] == datetime.fromisoformat(new_id.split("#")[0])) & (
                        ~pd.isna(aug_log.df[id_attribute])) & (aug_log.df[id_attribute] != "na")]
                # print(filtered[aug_log.timestamp], filtered[id_attribute])
                if len(filtered) > 0:
                    if len(filtered) > 1:
                        filtered = filtered.loc[filtered[aug_log.event_label].str.contains(new_id.split("#")[1])]
                        old_ids[new_id].update(
                            [str(v) for v in filtered[id_attribute].unique()])  # new_id.split("#")[1]
                    else:
                        old_ids[new_id].update(
                            [str(v) for v in filtered[id_attribute].unique()])  # new_id.split("#")[1]
        elif isinstance(new_id, str) and len(new_id.split("#")) == 4:
            id_attribute = [k for k, v in goldstandard["attibutes_to_obj_type"].items() if
                            new_id.split("#")[1] in v and goldstandard["attibutes_to_category"][k] == OBJ_INST]
            if len(id_attribute) > 0:
                id_attribute = id_attribute[0]
                if new_id.split("#")[1] not in old_ids:
                    old_ids[new_id] = set()  # new_id.split("#")[1]
                try:
                    fl_v = float(new_id.split("#")[3])
                except ValueError:
                    fl_v = new_id.split("#")[3]
                filtered = aug_log.df.loc[
                    (aug_log.df[aug_log.timestamp] == datetime.fromisoformat(new_id.split("#")[0])) & (
                        ~pd.isna(aug_log.df[id_attribute])) & (aug_log.df[id_attribute] != "na") & ((aug_log.df[
                                                                                                         aug_log.case_id] ==
                                                                                                     new_id.split("#")[
                                                                                                         3]) | (
                                                                                                            aug_log.df[
                                                                                                                aug_log.case_id] == fl_v))]
                if len(filtered) > 0:
                    old_ids[new_id].update(set(str(filtered.iloc[0][id_attribute])))
    return old_ids


def get_instance_event_scores(eval_result, base_ocel, gold_standard, ocel_log, aug_log, log, original_ocel=None,
                              only_correct_types=True):
    inst_event_tp = 0
    inst_event_fp = 0
    inst_event_fn = 0
    count = 0

    if original_ocel:
        original_events = ocel.get_events(original_ocel)
        print(len(original_events), len(ocel.get_events(ocel_log)))
        FN_PER_TYPE = {}
        FP_PER_TYPE = {}
        FN_PER_Event = {}
        FP_PER_Event = {}
        for event in log.events:
            if event.is_duplicate:
                continue
            old_from_new = get_old_id_from_new(new_event=event, aug_log=aug_log,
                                               goldstandard=gold_standard)
            omap_str = [str(o) for o in event.omap]
            original_event = original_events[str(event.vmap["EID"])]
            omap_to_consider = event.omap
            if only_correct_types:
                omap_to_consider = [oi for oi in omap_to_consider if log.type_mapping[oi] in log.object_types]

            for oinst in omap_to_consider:
                if log.type_mapping[oinst] not in gold_standard[OBJ_TYPE]:
                    continue
                # compute true positives and false positives
                count += 1
                if str(oinst) in original_event[OCEL_OMAP] or str(oinst) + ".0" in original_event[OCEL_OMAP] or (
                        isinstance(oinst, float) and str(int(oinst)) in original_event[OCEL_OMAP]) or (
                        oinst in old_from_new.keys() and any((old in original_event[OCEL_OMAP] or (
                        old + ".0") in original_event[OCEL_OMAP] or (old.replace(".0", "")) in original_event[
                                                                  OCEL_OMAP]) for old in
                                                             old_from_new[oinst])):
                    inst_event_tp = inst_event_tp + 1
                else:
                    inst_event_fp = inst_event_fp + 1
                    if log.type_mapping[oinst] not in FP_PER_TYPE:
                        FP_PER_TYPE[log.type_mapping[oinst]] = 1
                    else:
                        FP_PER_TYPE[log.type_mapping[oinst]] += 1
                    if event.label not in FP_PER_Event:
                        FP_PER_Event[event.label] = 1
                    else:
                        FP_PER_Event[event.label] += 1

            omap_to_consider = original_event[OCEL_OMAP]
            if only_correct_types:
                omap_to_consider = [oi for oi in omap_to_consider if obj_type_mapping[base_ocel][
                    ocel.get_objects(original_ocel)[oi][OCEL_TYPE]] in log.object_types]
            for oinst in omap_to_consider:
                # compute the false negatives
                all_old_new = set([val for k, v in old_from_new.items() for val in v])
                if oinst not in omap_str and oinst + ".0" not in omap_str and oinst.replace(".0",
                                                                                            "") not in omap_str and oinst.replace(
                    ".0", "") not in all_old_new and oinst not in all_old_new:
                    inst_event_fn = inst_event_fn + 1
                    if event.label not in FN_PER_Event:
                        FN_PER_Event[event.label] = 1
                    else:
                        FN_PER_Event[event.label] += 1

        print(FP_PER_TYPE)
        print(FN_PER_TYPE)
        print("#########")
        print(FP_PER_Event)
        print(FN_PER_Event)
        inst_event_precision = 0
        inst_event_recall = 0
        inst_event_f1 = 0
        if inst_event_tp > 0 or inst_event_fp > 0:
            inst_event_precision = inst_event_tp / (inst_event_tp + inst_event_fp)
        if inst_event_tp > 0 or inst_event_fn > 0:
            inst_event_recall = inst_event_tp / (inst_event_tp + inst_event_fn)
        if inst_event_precision > 0 or inst_event_recall > 0:
            inst_event_f1 = (2 * inst_event_precision * inst_event_recall) / (inst_event_recall + inst_event_precision)
        print("Instance-to-event precision", inst_event_precision)
        print("Instance-to-event recall", inst_event_recall)
        print("Instance-to-event f1", inst_event_f1)
        print("*" * 40)
        eval_result.inst_to_event_precision = inst_event_precision
        eval_result.inst_to_event_recall = inst_event_recall
        eval_result.inst_to_event_f1 = inst_event_f1
        return inst_event_precision, inst_event_recall, inst_event_f1, count
    else:
        print("REAL")
        return 0, 0, 0, 0


def evaluate_single_log(flat_name, base_ocel, ocel_name, config, gold_standard, gold_basic, ocel_log, aug_log, log,
                        instance_discoverer: InstanceDiscoverer, property_assigner: PropertyAssigner,
                        original_ocel=None, obj_type_occurrences=None):
    # The final evaluation result
    eval_result = EvaluationResult(log_name=flat_name, ocel_name=ocel_name, config=config)

    predictions = {}
    predictions[OBJ_TYPE] = list(ocel.get_object_types(ocel_log))
    attribute_name_to_pred_label = {}
    attribute_name_to_object = {}

    for eid, event in ocel.get_events(ocel_log).items():
        for att in event[OCEL_VMAP].keys():
            attribute_name_to_pred_label[att] = EVENT_ATT
    for oid, obj in ocel.get_objects(ocel_log).items():
        for att in obj[OCEL_OVMAP].keys():
            attribute_name_to_pred_label[att] = OBJ_PROP
            if att not in attribute_name_to_object:
                attribute_name_to_object[att] = {obj[OCEL_TYPE]}
            else:
                attribute_name_to_object[att].add(obj[OCEL_TYPE])
    for obj, att in instance_discoverer.object_to_id.items():
        attribute_name_to_object[att] = obj
    for att, cat in gold_standard["attibutes_to_category"].items():
        if att not in attribute_name_to_pred_label.keys():
            attribute_name_to_pred_label[att] = OBJ_INST
        if att not in attribute_name_to_object.keys():
            attribute_name_to_object[att] = "na"

    predictions["attibutes_to_category"] = attribute_name_to_pred_label
    predictions["attibutes_to_obj_type"] = attribute_name_to_object

    # calculating type level scores
    obj_type_precision, obj_type_recall, obj_type_f1, obj_type_count = get_object_type_scores(eval_result, predictions,
                                                                                              gold_standard, log)
    # calculating instance level scores
    obj_inst_precision_c, obj_inst_recall_c, obj_inst_f1_c, obj_inst_count_c = get_object_instance_scores(eval_result,
                                                                                                          predictions,
                                                                                                          gold_standard,
                                                                                                          ocel_log,
                                                                                                          aug_log, log,
                                                                                                          instance_discoverer,
                                                                                                          obj_type_occurrences=obj_type_occurrences)
    # calculating the property to instance scores
    inst_prop_precision_c, inst_prop_recall_c, inst_prop_f1_c, inst_prop_count_c = get_instance_prop_scores(eval_result,
                                                                                                            gold_standard,
                                                                                                            log,
                                                                                                            property_assigner,
                                                                                                            config)
    # calculating the instance to event scores, if possible (only applies to logs where we have a ground truth
    inst_event_precision_c, inst_event_recall_c, inst_event_f1_c, inst_event_count_c = get_instance_event_scores(
        eval_result, base_ocel, gold_standard, ocel_log, aug_log, log, original_ocel)

    eval_result.obj_type_precision = obj_type_precision
    eval_result.obj_type_recall = obj_type_recall
    eval_result.obj_type_f1 = obj_type_f1
    eval_result.obj_type_count = obj_type_count

    eval_result.obj_inst_precision = obj_inst_precision_c
    eval_result.obj_type_recall = obj_inst_recall_c
    eval_result.obj_inst_f1 = obj_inst_f1_c
    eval_result.obj_inst_count = obj_inst_count_c

    eval_result.obj_prop_precision = inst_prop_precision_c
    eval_result.obj_prop_recall = inst_prop_recall_c
    eval_result.obj_prop_f1 = inst_prop_f1_c
    eval_result.obj_prop_count = inst_prop_count_c

    eval_result.inst_to_event_precision = inst_event_precision_c
    eval_result.inst_to_event_recall = inst_event_recall_c
    eval_result.inst_to_event_f1 = inst_event_f1_c
    eval_result.inst_to_event_count = inst_event_count_c

    return eval_result


def write_results(eval_results):
    results_file = EVAL_OUTPUT_DIR + "results_" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(results_file, 'w', newline='') as csvfile:
        file_writer = csv.writer(csvfile, delimiter=';')
        header = ["config", "log", "ocelname", "explicit_ratio",
                  "obj_type_precision", "obj_type_recall", "obj_type_f1", "obj_type_count",
                  "obj_inst_precision", "obj_inst_recall", "obj_inst_f1", "obj_inst_count",
                  "obj_prop_precision", "obj_prop_recall", "obj_prop_f1", "obj_prop_count",
                  "inst_to_event_precision", "inst_to_event_recall", "inst_to_event_f1", "inst_to_event_count",
                  "time_s0", "time_s1", "time_s2", "time_s3", "time_s4", "time_s5"]
        file_writer.writerow(header)
        for result in eval_results:
            file_writer.writerow([result.config, result.log_name, result.ocel_name, result.explicit_ratio,
                                  result.obj_type_precision, result.obj_type_recall, result.obj_type_f1,
                                  result.obj_type_count,
                                  result.obj_inst_precision, result.obj_inst_recall, result.obj_inst_f1,
                                  result.obj_inst_count,
                                  result.obj_prop_precision, result.obj_prop_recall, result.obj_prop_f1,
                                  result.obj_prop_count,
                                  result.inst_to_event_precision, result.inst_to_event_recall, result.inst_to_event_f1,
                                  result.inst_to_event_count,
                                  result.time_s0, result.time_s1, result.time_s2, result.time_s3, result.time_s4,
                                  result.time_s5])

    return results_file


def set_times(res, old_res, timing):
    if 0 in timing:
        res.time_s0 = timing[0]
    elif old_res:
        res.time_s0 = old_res.time_s0
    if 1 in timing:
        res.time_s1 = timing[1]
    elif old_res:
        res.time_s1 = old_res.time_s1
    if 2 in timing:
        res.time_s2 = timing[2]
    elif old_res:
        res.time_s2 = old_res.time_s2
    if 3 in timing:
        res.time_s3 = timing[3]
    elif old_res:
        res.time_s3 = old_res.time_s3
    if 4 in timing:
        res.time_s4 = timing[4]
    elif old_res:
        res.time_s4 = old_res.time_s4
    if 5 in timing:
        res.time_s5 = timing[5]
    elif old_res:
        res.time_s5 = old_res.time_s5


def get_overall_results(results):
    obj_type_weighted_avg_precision = sum([res.obj_type_count * res.obj_type_precision for res in results]) / sum(
        [res.obj_type_count for res in results])
    obj_type_weighted_avg_recall = sum([res.obj_type_count * res.obj_type_recall for res in results]) / sum(
        [res.obj_type_count for res in results])
    obj_type_weighted_avg_f1 = sum([res.obj_type_count * res.obj_type_f1 for res in results]) / sum(
        [res.obj_type_count for res in results])
    obj_type_sum_count = sum([res.obj_type_count for res in results])

    obj_inst_weighted_avg_precision = sum([res.obj_inst_count * res.obj_inst_precision for res in results]) / sum(
        [res.obj_inst_count for res in results])
    obj_inst_weighted_avg_recall = sum([res.obj_inst_count * res.obj_inst_recall for res in results]) / sum(
        [res.obj_inst_count for res in results])
    obj_inst_weighted_avg_f1 = sum([res.obj_inst_count * res.obj_inst_f1 for res in results]) / sum(
        [res.obj_inst_count for res in results])
    obj_inst_sum_count = sum([res.obj_inst_count for res in results])

    obj_prop_weighted_avg_precision = sum([res.obj_prop_count * res.obj_prop_precision for res in results]) / sum(
        [res.obj_prop_count for res in results])
    obj_prop_weighted_avg_recall = sum([res.obj_prop_count * res.obj_prop_recall for res in results]) / sum(
        [res.obj_prop_count for res in results])
    obj_prop_weighted_avg_f1 = sum([res.obj_prop_count * res.obj_prop_f1 for res in results]) / sum(
        [res.obj_prop_count for res in results])
    obj_prop_sum_count = sum([res.obj_prop_count for res in results])

    inst_event_weighted_avg_precision = sum(
        [res.inst_to_event_count * res.inst_to_event_precision for res in results]) / sum(
        [res.inst_to_event_count for res in results])
    inst_event_weighted_avg_recall = sum([res.inst_to_event_count * res.inst_to_event_recall for res in results]) / sum(
        [res.inst_to_event_count for res in results])
    inst_event_weighted_avg_f1 = sum([res.inst_to_event_count * res.inst_to_event_f1 for res in results]) / sum(
        [res.inst_to_event_count for res in results])
    inst_event_sum_count = sum([res.inst_to_event_count for res in results])

    obj_type_avg_precision = mean([res.obj_type_precision for res in results])
    obj_type_avg_recall = mean([res.obj_type_recall for res in results])
    obj_type_avg_f1 = mean([res.obj_type_f1 for res in results])

    obj_inst_avg_precision = mean([res.obj_inst_precision for res in results])
    obj_inst_avg_recall = mean([res.obj_inst_recall for res in results])
    obj_inst_avg_f1 = mean([res.obj_inst_f1 for res in results])

    obj_prop_avg_precision = mean([res.obj_prop_precision for res in results])
    obj_prop_avg_recall = mean([res.obj_prop_recall for res in results])
    obj_prop_avg_f1 = mean([res.obj_prop_f1 for res in results])

    inst_event_avg_precision = mean([res.inst_to_event_precision for res in results])
    inst_event_avg_recall = mean([res.inst_to_event_recall for res in results])
    inst_event_avg_f1 = mean([res.inst_to_event_f1 for res in results])

    overall_precision = mean(
        [obj_type_weighted_avg_precision, obj_inst_weighted_avg_precision, obj_prop_weighted_avg_precision,
         inst_event_weighted_avg_precision])
    overall_recall = mean(
        [obj_type_weighted_avg_recall, obj_inst_weighted_avg_recall, obj_prop_weighted_avg_recall,
         inst_event_weighted_avg_recall])
    overall_f1 = mean(
        [obj_type_weighted_avg_f1, obj_inst_weighted_avg_f1, obj_prop_weighted_avg_f1,
         inst_event_weighted_avg_f1])
    overall_count = sum(
        [obj_type_sum_count, obj_inst_sum_count, obj_prop_sum_count,
         inst_event_sum_count])

    print(obj_type_sum_count, obj_type_avg_precision, obj_type_avg_recall, obj_type_avg_f1)
    print(obj_inst_sum_count, obj_inst_avg_precision, obj_inst_avg_recall, obj_inst_avg_f1)
    print(obj_prop_sum_count, obj_prop_avg_precision, obj_prop_avg_recall, obj_prop_avg_f1)
    print(inst_event_sum_count, inst_event_avg_precision, inst_event_avg_recall, inst_event_avg_f1)
    # print(overall_count, overall_precision, overall_recall, overall_f1)
    print("-" * 40)
    print(obj_type_sum_count, obj_type_weighted_avg_precision, obj_type_weighted_avg_recall, obj_type_weighted_avg_f1)
    print(obj_inst_sum_count, obj_inst_weighted_avg_precision, obj_inst_weighted_avg_recall, obj_inst_weighted_avg_f1)
    print(obj_prop_sum_count, obj_prop_weighted_avg_precision, obj_prop_weighted_avg_recall, obj_prop_weighted_avg_f1)
    print(inst_event_sum_count, inst_event_weighted_avg_precision, inst_event_weighted_avg_recall,
          inst_event_weighted_avg_f1)
    print(overall_count, overall_precision, overall_recall, overall_f1)

    return obj_type_sum_count, obj_type_weighted_avg_precision, obj_type_weighted_avg_recall, \
           obj_type_weighted_avg_f1, obj_inst_sum_count, obj_inst_weighted_avg_precision, \
           obj_inst_weighted_avg_recall, obj_inst_weighted_avg_f1, obj_prop_sum_count, \
           obj_prop_weighted_avg_precision, obj_prop_weighted_avg_recall, obj_prop_weighted_avg_f1, \
           inst_event_sum_count, inst_event_weighted_avg_precision, inst_event_weighted_avg_recall, \
           inst_event_weighted_avg_f1, overall_precision, overall_recall, overall_f1, overall_count


DO_MASKING = False
REAL = False
ALL = True

# random name for atts
NO_NAME = False

def run_eval(conf):
    results = []
    for (dir_path, dir_names, filenames) in os.walk(conf.input_dir + "raw/"):
        for filename in filenames:
            print(filename)
            if filename.endswith('.xes') or filename.endswith('.csv'):
                if REAL:
                    if "19" not in filename and "17" not in filename:
                        continue
                else:
                    if "item" not in filename and "package" not in filename and "order" not in filename:
                        continue
                if NO_NAME and "nn." not in filename:
                    continue
                if not NO_NAME and "nn." in filename:
                    continue
                # We need to use the models that were trained without the current data,
                # this is handled by setting the current run to the raw file
                current_run = set_current_run(filename)
                obj_type_occurrences = None
                original_ocel = None
                if current_run in ocel_logs:
                    original_ocel = loader.load_ocel(dir_path, current_run)
                    obj_type_occurrences = {obj_type_mapping[current_run][obj_type]: set() for obj_type in
                                            ocel.get_object_types(original_ocel)}
                    for eid, ev in ocel.get_events(original_ocel).items():
                        for oinst in ev[OCEL_OMAP]:
                            obj_type_occurrences[
                                obj_type_mapping[current_run][ocel.get_objects(original_ocel)[oinst][OCEL_TYPE]]].add(
                                oinst)
                    for obj_type, oiset in obj_type_occurrences.items():
                        obj_type_occurrences[obj_type] = len(oiset)
                current_config = deepcopy(conf)
                try:
                    current_config.model_dir = current_config.model_dir + str(model_idx[current_run]) + '/'
                except ValueError:
                    print(filename, "not in evaluation logs")
                current_config.exclude_data_origin.append(current_run)

                if not DO_MASKING:
                    new_ocel_name, log, aug_log, instance_discoverer, property_assigner, instance_assigner, timing = augment_and_transform_log(
                        dir_path, filename, current_config)
                    if original_ocel is None:
                        continue
                    gold_standard = get_gold_standard(
                        os.path.join(dir_path.replace("raw", "gold"), filename + "_gs.json"))
                    ocel_log = loader.load_ocel(current_config.output_dir, new_ocel_name)
                    old_res = loader.load_previous_result(current_config.output_dir, current_config, filename)
                    res = evaluate_single_log(filename, current_run, new_ocel_name, current_config, gold_standard,
                                              gold_standard, ocel_log, aug_log, log, instance_discoverer,
                                              property_assigner,
                                              original_ocel=original_ocel, obj_type_occurrences=obj_type_occurrences)
                    set_times(res, old_res, timing)
                    writer.serialize_eval_result(current_config.output_dir, current_config, filename, res)
                    results.append(res)

                # sys.exit(0)
                if DO_MASKING and not REAL and not ALL:
                    gold_standard = get_gold_standard(
                        os.path.join(dir_path.replace("raw", "gold"), filename + "_gs.json"))
                    for att, cat in gold_standard["attibutes_to_category"].items():
                        if cat == OBJ_INST and att != "case:concept:name":  # and att != log.case_id:# and att == "OfferID":
                            if att not in ["items", "orders", "packages"]:
                                continue
                            print("Ok, it is getting serious, we mask instance attribute", att, "now")
                            new_conf = deepcopy(current_config)
                            new_conf.mask_attribute.append(att)
                            new_ocel_name, log, aug_log, instance_discoverer, property_assigner, instance_assigner, timing = augment_and_transform_log(
                                dir_path, filename, new_conf)
                            if original_ocel is None:
                                continue
                            ocel_log = loader.load_ocel(current_config.output_dir, new_ocel_name)
                            old_res = loader.load_previous_result(current_config.output_dir, current_config, filename)
                            res = evaluate_single_log(filename, current_run, new_ocel_name, new_conf,
                                                      gold_standard, gold_standard, ocel_log, aug_log, log,
                                                      instance_discoverer, property_assigner,
                                                      original_ocel=original_ocel,
                                                      obj_type_occurrences=obj_type_occurrences)
                            set_times(res, old_res, timing)
                            writer.serialize_eval_result(new_conf.output_dir, new_conf, filename, res)
                            results.append(res)
                if DO_MASKING and not REAL and ALL:
                    gold_standard = get_gold_standard(
                        os.path.join(dir_path.replace("raw", "gold"), filename + "_gs.json"))

                    atts = []
                    for att, cat in gold_standard["attibutes_to_category"].items():

                        if cat == OBJ_INST and att != "case:concept:name":
                            # if att not in ["items", "orders", "packages"]:
                            #    continue
                            atts.append(att)
                    print("Ok, it is getting serious, we mask instance attribute", atts, "now")
                    new_conf = deepcopy(current_config)
                    for att in atts:
                        new_conf.mask_attribute.append(att)
                    new_ocel_name, log, aug_log, instance_discoverer, property_assigner, instance_assigner, timing = augment_and_transform_log(
                        dir_path, filename, new_conf)
                    if original_ocel is None:
                        continue
                    ocel_log = loader.load_ocel(current_config.output_dir, new_ocel_name)
                    old_res = loader.load_previous_result(current_config.output_dir, current_config, filename)
                    res = evaluate_single_log(filename, current_run, new_ocel_name, new_conf,
                                              gold_standard, gold_standard, ocel_log, aug_log, log,
                                              instance_discoverer, property_assigner,
                                              original_ocel=original_ocel,
                                              obj_type_occurrences=obj_type_occurrences)
                    set_times(res, old_res, timing)
                    writer.serialize_eval_result(new_conf.output_dir, new_conf, filename, res)
                    results.append(res)
    if len(results) > 0:
        get_overall_results(results)
        write_results(results)


if __name__ == '__main__':
    conf = DEFAULT_CONFIG
    main_tic = time.perf_counter()
    run_eval(conf)
    main_toc = time.perf_counter()
    print(f"Program finished all operations in {main_toc - main_tic:0.4f} seconds")
    sys.exit()
