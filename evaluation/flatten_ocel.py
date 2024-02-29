from copy import deepcopy
from random import random

import ocel
from const import *
import csv
import os

def flatten_log(ocel_log, dir, name, bo, mask_instance=0):
    import pandas as pd
    attributes = list(ocel.get_attribute_names(ocel_log))
    attributes.append("EID")
    covered_object_types = {bo}
    objects = list(ocel.get_object_types(ocel_log))
    objects.remove(bo)
    # We allow to randomly mask the instance and property information of particular object types
    masks = set()
    if mask_instance > 0:
        masks.update(random.choices(objects, k=mask_instance))
    cases = {}

    act_label_to_obj_types = {}
    for oid, objty in ocel.get_objects(ocel_log).items():
        if objty[OCEL_TYPE] == bo:
            cases[oid] = []
            for eid, event in ocel.get_events(ocel_log).items():
                if oid in event[OCEL_OMAP]:
                    event[OCEL_VMAP]["EID"] = eid
                    cases[oid].append(event)
    for eid, event in ocel.get_events(ocel_log).items():
        if event[OCEL_ACTIVITY] not in act_label_to_obj_types:
            act_label_to_obj_types[event[OCEL_ACTIVITY]] = set()
        objs = {obj: 0 for obj in objects}
        for oid in event[OCEL_OMAP]:
            o = ocel.get_objects(ocel_log)[oid][OCEL_TYPE]
            if o == bo:
                continue
            objs[o] += 1
        for obj, num in objs.items():
            if num != 1:
                act_label_to_obj_types[event[OCEL_ACTIVITY]].add(obj)

    print(len(cases))
    flat_name = name + bo + ".csv"
    flat_name = flat_name.replace(".jsonocel", "")
    with open(os.path.join(dir, flat_name), 'w', newline='') as csvfile:
        fieldnames = [XES_CASE, XES_TIME, XES_NAME] + [obj for obj in objects if
                                                       obj not in masks] + attributes
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for cid, case in cases.items():
            obj_type_to_inst = {}
            for event in case:
                objs = {}
                for obj in event[OCEL_OMAP]:
                    ty = ocel.get_objects(ocel_log)[obj][OCEL_TYPE]
                    if ty != bo:
                        if ty not in objs.keys():
                            objs[ty] = []
                        objs[ty].append(obj)
                objs = {obj_ty: obj_inst[0] for obj_ty, obj_inst in objs.items() if
                        obj_ty not in act_label_to_obj_types[event[OCEL_ACTIVITY]]}
                covered_object_types.update(objs.keys())
                objs_atts = {obj_att: val for obj_ty, obj_inst in objs.items() for obj_att, val in
                             ocel.get_objects(ocel_log)[obj_inst][OCEL_OVMAP].items() if
                             obj_ty not in act_label_to_obj_types[event[OCEL_ACTIVITY]]}
                # atts = {ocel.get_objects(ocel_log)[obj][OCEL_TYPE]: obj for att in attributes}
                for o in objects:
                    if o not in objs.keys() and o not in masks:
                        objs[o] = ""
                atts = {att: event[OCEL_VMAP][att] if att in event[OCEL_VMAP].keys() else "" for att in attributes}
                atts.update(objs_atts)
                to_write = {XES_CASE: cid, XES_NAME: event[OCEL_ACTIVITY],
                            XES_TIME: event[OCEL_TIMESTAMP]}
                to_write.update(objs)
                to_write.update(atts)
                writer.writerow(to_write)

    df = pd.read_csv(os.path.join(dir, flat_name), sep=";")
    to_remove = []
    for col in df.columns:
        if pd.isnull(df[col]).all():
            to_remove.append(col)
    for col in to_remove:
        df.drop(col, 1, inplace=True)
        if col in covered_object_types:
            covered_object_types.remove(col)
    df.to_csv(os.path.join(dir, flat_name), sep=";")
    return flat_name, covered_object_types


def flatten_pm4py(ocel_log, bo, name, direct):
    import pm4py
    from pm4py.objects.conversion.log import converter
    covered_object_types = {bo}
    flat_name = name + bo + "pm4py.csv"
    flat_name = flat_name.replace(".jsonocel", "")
    flattened_log = pm4py.ocel_flattening(ocel_log, bo)
    df = converter.apply(flattened_log, variant=converter.Variants.TO_DATA_FRAME)
    df.to_csv(os.path.join(direct, flat_name), sep=";")
    return flat_name, covered_object_types
