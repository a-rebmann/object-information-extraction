from collections import OrderedDict

from model.augmented_log import AugmentedLog
from model.log.case import Case
from model.log.event import Event
import pandas as pd
from const import *
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from preprocessing.nlp_utils import nlp

nouns = {noun for x in wn.all_synsets('n') for noun in x.name().split('.', 1)}

from preprocessing.preprocessor import clean_attribute_name, clean_attribute_name_for_object


class Log(object):

    def __init__(self, aug_log, cases: dict, name='default', original_atts=None, attribute_names=None,
                 att_names_no_case_atts=None,
                 object_types=None, obj_types_from_event_atts=None, check_duplicate=True, obj_att_name_matches=None,
                 duplication_atts_per_obj=None,
                 duplication_dict=None, dup_atts=None, object_from_case=None):
        self.aug_log = aug_log
        self.cases = cases
        self.num_cases = len(cases)
        self.events = [e for case in self.cases.values() for e in case.events]
        self.events.sort(key=lambda x: x.timestamp, reverse=False)
        self.name = name
        self.case_id = 'case:' + XES_NAME
        self.event_label = OCEL_ACTIVITY
        self.timestamp = OCEL_TIMESTAMP
        self._traces = None
        self.objects = dict()
        self.object_types = set() if not object_types else object_types
        self.attribute_names = set() if not attribute_names else attribute_names
        self.att_names_no_case_atts = set() if not att_names_no_case_atts else att_names_no_case_atts
        self.type_mapping = dict()
        self.obj_types_from_event_atts = set() if not obj_types_from_event_atts else obj_types_from_event_atts
        self.objs_per_type = None
        self.check_duplicate = check_duplicate
        self.obj_att_name_matches = {} if not obj_att_name_matches else obj_att_name_matches
        self.lemmatizer = WordNetLemmatizer()
        self.cleaned_name = [self.lemmatizer.lemmatize(tok) for tok in clean_attribute_name(self.name).split(" ")]
        self.case_object = None
        self.duplication_atts_per_obj = duplication_atts_per_obj
        self.original_atts_to_consider = original_atts
        self.selected_attributes = [x for x in self.original_atts_to_consider if
                                    self.aug_log.attribute_to_concept_type[x] == type_mapping[
                                        ConceptType.BO_NAME.value] or
                                    self.aug_log.attribute_to_concept_type[x] == ConceptType.OTHER.value or
                                    self.aug_log.attribute_to_concept_type[x] == type_mapping[
                                        ConceptType.BO_INSTANCE.value] or
                                    self.aug_log.attribute_to_concept_type[x] == type_mapping[
                                        ConceptType.BO_PROP.value] or
                                    self.aug_log.attribute_to_concept_type[x] == type_mapping[
                                        ConceptType.ACTOR_INSTANCE.value] or
                                    self.aug_log.attribute_to_concept_type[x] == ConceptType.OTHER.value]
        self.duplication_dict = dict() if not duplication_dict else duplication_dict
        self.dup_atts = dup_atts
        self.object_from_case = object_from_case

    def find_duplicated_event_old(self, other: Event) -> list:
        duplicates = []
        for event in self.events:
            if event.is_duplicate:
                continue
            if event.label == other.label and event.timestamp == other.timestamp and \
                    all((att not in event.vmap and att not in other.vmap) or (
                            att in event.vmap and att in other.vmap and event.vmap[att] == other.vmap[att]) for att in
                        self.att_names_no_case_atts):
                duplicates.append(event)
        return duplicates

    def find_duplicates(self, event):
        duplication_atts = self.dup_atts
        for obj_typ in event.object_type_to_actions.keys():
            if obj_typ in self.duplication_atts_per_obj:
                duplication_atts = self.duplication_atts_per_obj
                break
        dupl_id = str(event.label) + str(event.timestamp) + str(
            [event.vmap[att] for att in event.vmap.keys() if att in duplication_atts])
        if dupl_id in self.duplication_dict:
            return self.duplication_dict[dupl_id]
        else:
            return set()

    @property
    def traces(self):
        if self._traces is None:
            self._traces = [c.trace for c in self.cases.values()]
        return self._traces

    @property
    def objects_per_type(self):
        if self.objs_per_type is None:
            self.objs_per_type = {obj_ty: set() for obj_ty in self.object_types}
            for obj_id, obj in self.objects.items():
                try:
                    self.objs_per_type[obj.obj_type].add(obj)
                except KeyError:
                    print(obj_id, obj)
        return self.objs_per_type

    def check_cardinalities(self, instance_discoverer):
        print(instance_discoverer.cardinality)
        return instance_discoverer.cardinality

        # TODO replace with check for N:M relations
        # if "running" in self.name:
        #     return N_TO_M
        # elif "19" in self.name:
        #     return N_TO_ONE
        # else:
        #     return ONE_TO_N

    def check_uniqueness_across_cases(self, att, obj_ty=None):
        if self.aug_log.case_id == att:
            return 1.0
        filtered = self.aug_log.df.loc[
            (~self.aug_log.df["con_keep"]) & (~pd.isnull(self.aug_log.df[att])) & (
                    self.aug_log.df[att] != "na")]
        for lab, group in filtered.groupby(self.aug_log.event_label):
            by_case = group.groupby(self.aug_log.case_id).first()
            if ((len(by_case[att].unique()) / len(by_case[att]) > 0.9) and len(by_case[att]) >= len(
                    self.cases) / 10) or (
                    (len(by_case[att].unique()) / len(by_case[att]) > 0.9) and obj_ty in clean_attribute_name(lab)):
                print(att, "unique for label ", lab)
                return 1.0
        one_per_case_no_duplicates = filtered.groupby(self.aug_log.case_id).first()
        if len(one_per_case_no_duplicates[att]) == 0:
            return 0
        uniqueness = len(one_per_case_no_duplicates[att].unique()) / len(one_per_case_no_duplicates[att])
        # print(uniqueness, "uniqueness", att)
        return uniqueness

    def check_uniqueness_across_cases_obj(self, obj_ty, att):
        if self.aug_log.case_id == att:
            return 1.0
        filtered = self.aug_log.df.loc[
            (~self.aug_log.df["con_keep"]) & (~pd.isnull(self.aug_log.df[att])) & (
                    self.aug_log.df[att] != "na")]
        for lab, group in filtered.groupby(self.aug_log.event_label):
            if obj_ty not in clean_attribute_name(lab):
                continue
            by_case = group.groupby(self.aug_log.case_id).first()
            if (len(by_case[att].unique()) / len(by_case[att]) > 0.99) and len(by_case[att]) >= len(self.cases) / 5:
                print(att, "unique for label ", lab)
                return 1.0
        one_per_case_no_duplicates = filtered.groupby(self.aug_log.case_id).first()
        if len(one_per_case_no_duplicates[att]) == 0:
            return 0
        uniqueness = len(one_per_case_no_duplicates[att].unique()) / len(one_per_case_no_duplicates[att])
        # print(uniqueness, "uniqueness", att)
        return uniqueness

    def check_evenness_thresh(self, att, thresh=.5):
        if self.aug_log.case_id == att:
            return True
        unique_vals = self.aug_log.df[att].dropna().unique()
        min_u = min(unique_vals)
        max_u = max(unique_vals)
        return ((max_u - min_u) * thresh) <= (len(unique_vals) - 1)

    def isint(self, att):
        return self.aug_log.attribute_to_attribute_type[att] == AttributeType.INT

    def isnumeric(self, att):
        return (self.aug_log.attribute_to_attribute_type[att] == AttributeType.NUMERIC or
                self.aug_log.attribute_to_attribute_type[att] == AttributeType.INT)

    def isdefinitelynotid(self, att):
        return (self.aug_log.attribute_to_attribute_type[att] == AttributeType.FLAG or
                self.aug_log.attribute_to_attribute_type[att] == AttributeType.TIMESTAMP)

    def isstring(self, att):
        return (self.aug_log.attribute_to_attribute_type[att] == AttributeType.STRING)


def check_proper_object(tok):
    prop = len(tok) > 2 and tok in words.words() and (
            tok in nouns or tok in SCHEMA_ORG_OBJ) and tok not in COMMON_ABBREV
    # print(prop, tok)
    return prop


def find_objects(split, tags):
    res = [tok for tok, bo in zip(split, tags) if
           (bo == 'BO' or (bo == "X" and tok in stopwords.words('english'))) and (
               check_proper_object(tok))]
    if len(res) > 0 and res[0] in stopwords.words('english'):
        res = res[1:]
    if len(res) > 0 and res[-1] in stopwords.words('english'):
        res = res[0:-1]
    return res


def find_actions(split, tags):
    return [tok for tok, a in zip(split, tags) if a in ['A', 'STATE']]


def check_att(aug_log, att, obj, att_dict, all_objects_from_labels):
    # this checks for case attribute, i.e., attributes that are constant for all events in a case for all cases,
    # and object attributes
    if "case:" in att:
        s = 1
    else:
        if att in att_dict:
            s = att_dict[att]
        else:
            s = sum(len(current_case[att].dropna().unique()) == 1 for case_id, current_case in aug_log.cases) / len(
                aug_log.cases)
            att_dict[att] = s
    if obj:
        nlp_check = (obj in clean_attribute_name(att) or clean_attribute_name_for_object(att) in obj or any(
            nlp(word)[0].lemma_ in [nlp(token)[0].lemma_ for token in
                                    clean_attribute_name_for_object(att).split(" ") if
                                    len(nlp(token)) > 0] for word in obj.split(" ") if len(nlp(word))))
    else:
        for obj1 in all_objects_from_labels:
            if obj1 != nlp(clean_attribute_name(att))[0].lemma_ and obj1 in clean_attribute_name(att) and not (
                    aug_log.attribute_to_attribute_type[att] == AttributeType.FLAG or
                    aug_log.attribute_to_attribute_type[att] == AttributeType.TIMESTAMP):
                return True
        nlp_check = False
    return s < .5 or nlp_check


def from_pd_log(aug_log: AugmentedLog, name, config):
    """
    :return: a custom log representation
    """
    lemmatizer = WordNetLemmatizer()
    nlp.get_pipe("lemmatizer")

    event_duplication_dict = {}

    actions_for_label = {label: find_actions(aug_log.tagged_labels[label][0],
                                             aug_log.tagged_labels[label][1]) for label in aug_log.tagged_labels.keys()}
    objects_for_label = {label: lemmatizer.lemmatize(" ".join(find_objects(aug_log.tagged_labels[label][0],
                                                                           aug_log.tagged_labels[label][1]))) for label
                         in aug_log.tagged_labels.keys()}
    obj_for_lab = {}
    for lab, ob in objects_for_label.items():
        obj_for_lab[lab] = ob
    objects_for_label = obj_for_lab

    all_objects_from_labels = set([ob for ob in objects_for_label.values()])
    if '' in all_objects_from_labels:
        all_objects_from_labels.remove('')
    #print(all_objects_from_labels)
    #print(len(aug_log.df))
    object_from_case = dict()
    duplicates = {}
    for obj_type in aug_log.case_level_bo:
        if len(obj_type) <= 1:
            continue
        # HANDLE NAME MATCHES
        if not any(tok in all_objects_from_labels for tok in obj_type.split(" ")):
            object_from_case[obj_type] = aug_log.case_bo_to_case_att[obj_type]
            if obj_type not in duplicates:
                duplicates[obj_type] = set()
            duplicates[obj_type].add(aug_log.case_bo_to_case_att[obj_type])
        elif any(tok in all_objects_from_labels for tok in obj_type.split(" ")):
            if obj_type not in duplicates:
                duplicates[obj_type] = set()
            duplicates[obj_type].add(aug_log.case_bo_to_case_att[obj_type])
        elif any(tok in all_objects_from_labels for tok in
                 clean_attribute_name(aug_log.case_bo_to_case_att[obj_type]).split(" ")):
            plain = clean_attribute_name(aug_log.case_bo_to_case_att[obj_type])

            ex_obj_type = [ob for ob in all_objects_from_labels if ob in plain.split(" ")][0]
            if ex_obj_type not in duplicates:
                duplicates[ex_obj_type] = set()
            duplicates[ex_obj_type].add(aug_log.case_bo_to_case_att[obj_type])
    original_atts = [att for att in aug_log.attribute_to_attribute_type.keys() if
                     att not in config.mask_attribute]
    att_dict = dict()
    # remove our artificial attribute from consideration, since this is not part of the original OCEL log!
    if "EID" in original_atts:
        original_atts.remove("EID")
    sub_1 = [k for k, v in aug_log.attribute_to_attribute_type.items() if
             k in original_atts and check_att(aug_log, k, None, att_dict, all_objects_from_labels)
             and k != aug_log.case_id]
    #print(sub_1, "filter")
    m1 = aug_log.df.duplicated(subset=sub_1, keep="first")
    aug_log.df['con_keep'] = m1
    dup_atts = sub_1
    duplication_atts_per_obj = {}
    for obj in all_objects_from_labels:
        sub = [k for k, v in aug_log.attribute_to_attribute_type.items() if
               k in original_atts and check_att(aug_log, k, obj, att_dict, all_objects_from_labels)
               and k != aug_log.case_id]
        duplication_atts_per_obj[obj] = sub

        if len(sub) < 1 or set(sub).issubset(set(sub_1)):
            duplication_atts_per_obj[obj] = sub_1
            continue
        else:
            duplication_atts_per_obj[obj] = sub
        #print(obj, sub)
        m2 = aug_log.df.duplicated(subset=sub, keep="first")
        aug_log.df.loc[aug_log.df[aug_log.event_label].str.contains(obj, case=False, na=False), "con_keep"] = m2
    aug_log.df.to_csv(config.output_dir + name, sep=';')

    if aug_log.timestamp is not None:
        grouped = aug_log.df.sort_values([aug_log.case_id, aug_log.timestamp],
                                         ascending=[True, True]).groupby(aug_log.case_id)
    else:
        grouped = aug_log.df.sort_values([aug_log.case_id], ascending=[True]).groupby(
            aug_log.case_id)
    # all attribute names that can be considered
    att_names = [col for col in aug_log.attribute_to_attribute_type.keys() if
                 col != aug_log.event_label and col != aug_log.timestamp]
    # attribute names excluding case attributes
    att_names_no_case_atts = [k for k, v in aug_log.attribute_to_attribute_type.items() if
                              k not in aug_log.case_attributes and k != aug_log.case_id]

    obj_types_from_event_atts = set()
    object_types = set()
    cases = OrderedDict()
    check_duplicate = True
    try:
        if all(aug_log.df[aug_log.timestamp].dt.minute == 0) and all(aug_log.df[aug_log.timestamp].dt.second == 0):
            check_duplicate = False
            print("do not check for duplicates due to timestamp granularity issue")
    except AttributeError:
        print("do not check for duplicates due to timestamp format issue")
    for case_id, case in grouped:
        events = []
        for index, row in case.iterrows():
            label = row[aug_log.event_label]
            timestamp = row[aug_log.timestamp]
            vmap = {col: row[col] for col in aug_log.attribute_to_attribute_type.keys() if
                    col != aug_log.event_label and col != aug_log.timestamp and not pd.isna(row[col]) and not row[
                                                                                                                  col] in TERMS_FOR_MISSING}
            event = Event(label=label, timestamp=timestamp, vmap=vmap)
            if row["con_keep"] and check_duplicate:
                event.is_duplicate = True
            obj_type_to_actions = {}
            # extract objects types and actions
            action_in_label = False
            label_obj = None
            label_acts = []

            if aug_log.event_label + "_cleaned" in aug_log.df.columns and row[
                aug_log.event_label + "_cleaned"] in aug_log.tagged_labels:
                actions = actions_for_label[row[aug_log.event_label + "_cleaned"]]
                obj_type = objects_for_label[row[aug_log.event_label + "_cleaned"]]
                if len(actions) > 0:
                    action_in_label = True
                    if obj_type != "":
                        obj_type_to_actions[obj_type] = actions
                        object_types.add(obj_type)
                        obj_types_from_event_atts.add(obj_type)
                    else:
                        label_acts = actions
                elif obj_type != "":
                    label_obj = obj_type
            # extract object types from other textual attributes
            for obj_type in object_from_case.keys():
                if not pd.isna(row[aug_log.case_bo_to_case_att[obj_type]]) and not row[aug_log.case_bo_to_case_att[
                    obj_type]] in TERMS_FOR_MISSING:
                    obj_type_to_actions[obj_type] = []
                    object_types.add(obj_type)
            for att in [att for att in aug_log.get_attributes_by_att_types(consider_for_tagging)]:
                if att + "_cleaned" in aug_log.df.columns and row[att + "_cleaned"] in aug_log.tagged_labels:
                    try:
                        obj_type = objects_for_label[row[att + "_cleaned"]]
                        # print(obj_type)
                    except KeyError:
                        print(aug_log.event_label + "_cleaned", "not in DF")
                        obj_type = ""
                    if obj_type != "" and obj_type not in obj_type_to_actions:

                        obj_type_to_actions[obj_type] = [label_act for label_act in label_acts]
                        label_acts = []
                        object_types.add(obj_type)
                        if att not in aug_log.case_attributes:
                            obj_types_from_event_atts.add(obj_type)
                    if not action_in_label and label_obj:
                        actions = actions_for_label[row[aug_log.event_label + "_cleaned"]]
                        if len(actions) > 0:
                            obj_type_to_actions[label_obj] = actions
                            action_in_label = True
            event.object_type_to_actions = obj_type_to_actions
            events.append(event)
            if event.is_duplicate:
                duplication_atts = dup_atts
                for obj_typ in event.object_type_to_actions.keys():
                    if obj_typ in duplication_atts_per_obj:
                        duplication_atts = duplication_atts_per_obj
                        break
                dupl_id = str(event.label) + str(event.timestamp) + str(
                    [event.vmap[att] for att in event.vmap.keys() if att in duplication_atts])
                if dupl_id not in event_duplication_dict:
                    event_duplication_dict[dupl_id] = set()
                event_duplication_dict[dupl_id].add(event)
        case = Case(case_id=case_id, events=events)
        cases[case_id] = case
    #print(object_types)
    return Log(aug_log=aug_log, cases=cases, name=name, original_atts=original_atts, attribute_names=att_names,
               att_names_no_case_atts=att_names_no_case_atts,
               object_types=object_types, obj_types_from_event_atts=obj_types_from_event_atts,
               check_duplicate=check_duplicate, obj_att_name_matches=duplicates,
               duplication_atts_per_obj=duplication_atts_per_obj, duplication_dict=event_duplication_dict,
               dup_atts=dup_atts, object_from_case=object_from_case)
