from const import AttributeType, ConceptType, type_mapping, consider_for_tagging, consider_for_label_classification, \
    XES_NAME, XES_TIME, UNIQUE_EVENT_ID
from preprocessing.nlp_utils import TERMS_FOR_MISSING, get_pos
import pandas as pd
import itertools
import numpy as np
from pm4py.objects.conversion.log import converter


def find_components(split, tags):
    labels = []
    curr = ''
    pos = get_pos(' '.join(split))
    for tok, bo in zip(split, tags):
        if bo == ConceptType.ACTION_NAME.value or bo == ConceptType.BO_NAME.value:
            curr = " ".join([curr, tok])
        if bo != ConceptType.BO_NAME.value and bo != ConceptType.ACTION_NAME.value and curr != '':
            if bo == ConceptType.OTHER.value:
                if pos[split.index(tok)] == 'CONJ':
                    labels.append(curr)
                    curr = ''
    if curr != '':
        labels.append(curr)
    return labels


def find_role(split, tags, role):
    roles = []
    curr = ''
    for tok, r in zip(split, tags):
        if r == role:
            curr = " ".join([curr, tok])
        if r != role and curr != '':
            roles.append(curr.strip())
            curr = ''
    if curr != '':
        roles.append(curr.strip())
    return roles


class AugmentedLog:

    def __init__(self, name, df: pd.DataFrame, case, event_label=None, timestamp=None):
        self.name = name
        # the raw log as a data frame
        self.df = df
        # the preprocessed log, i.e, strings are tokenized and free of digits and special chars
        self.cleaned_df = pd.DataFrame()
        # the augmented data frame
        self.augmented_df = None
        # attributes whose values are identical for the entire log
        self.duplicates = {}
        # a dictionary to store the attributes and their type (e.g., rich text vs timestamp vs numeric...)
        self.attribute_to_attribute_type = {}
        # case attributes are not considered right now and have their own 'type'
        self.attribute_to_attribute_type[case] = AttributeType.CASE
        # a dictionary to store the attributes and their assigned role if any
        self.attribute_to_concept_type = {}
        # an exception is the case id which is vacuously assigned the business object instance role
        self.attribute_to_concept_type[case] = ConceptType.BO_INSTANCE.value
        # a dictionary to store the attributes and their type (e.g., rich text vs timestamp vs numeric...)
        self.attribute_to_ne = {}
        # cluttered rich text labels mapped to cleaned ones (like in cleaned_df)
        self.cleaned_labels = {}
        # cleaned rich text labels mapped to the tokens and corresponding role tags that were assigned to them
        self.tagged_labels = {}
        # labels that contain named entities
        self.label_to_ne = {}
        # resource values and their corresponding type, i.e., whether they are human or another resource (machine,
        # system, ...)
        self.resource_to_type = {}
        # attribute names mapped to their unique values
        self.att_to_unique = {}
        # attribute names mapped to the number of unique values they cover
        self.num_uniques = {}
        # attribute names mapped to a list of samples of their values
        self.samples = {}
        # a list of case attributes
        self.case_attributes = list()
        # the case id, which has to be known due to technical reasons (PM4Py requires it)
        self.case_id = case
        # the timestamp attribute indicating the point in time when the event occurred
        self.event_label = event_label if event_label is not None else XES_NAME
        # the timestamp attribute indicating the point in time when the event occurred
        self.timestamp = timestamp if timestamp is not None else XES_TIME
        # newly created labels after label augmentation (specific use case)
        self.augmented = {}
        # a grouping by cases
        self.cases = None
        # a set of object types that occur in all events
        self.case_level_bo = set()
        # and a mappting to remeber where an object type came from
        self.case_bo_to_case_att = dict()
        # filling the necessary values to some of the above defined maps
        self._initialize_maps()

    def get_num_cases(self):
        return len(self.df[self.case_id].unique())

    def add_ne_for_label(self, label, list_for_nes):
        self.label_to_ne[label] = list_for_nes

    def add_tagged_vals(self, tagged: dict):
        for key, tags in tagged.items():
            if key not in self.tagged_labels.keys():
                self.tagged_labels[key] = tags

    def add_tagged_label(self, label, tags, old_label):
        if label not in self.tagged_labels.keys():
            self.tagged_labels[label] = tags
            self.augmented[old_label] = label

    def get_tags_for_label(self, label: str):
        if label in self.tagged_labels.keys():
            return self.tagged_labels[label]
        return None

    def set_attribute_type(self, attribute: str, attribute_type: AttributeType):
        if AttributeType.CASE == attribute_type:
            print("CANNOT RESET CASE ID!")
            return
        self.attribute_to_attribute_type[attribute] = attribute_type

    def set_concept_type(self, attribute: str, concept_type: ConceptType):
        if type(concept_type) is ConceptType:
            self.attribute_to_concept_type[attribute] = concept_type.value
        else:
            self.attribute_to_concept_type[attribute] = concept_type

    def get_attribute_annotation(self):
        return self.attribute_to_attribute_type.copy()

    def get_concept_annotation(self):
        return self.attribute_to_concept_type.copy()

    def get_attributes_by_att_type(self, att_type: AttributeType):
        filtered = []
        for att in self.attribute_to_attribute_type.keys():
            if self.attribute_to_attribute_type[att] == att_type:
                filtered.append(att)
        return filtered

    def get_attributes_by_role(self, role: AttributeType):
        atts = []
        for att in self.attribute_to_concept_type.keys():
            if self.attribute_to_concept_type[att] == role:
                atts.append(att)
        return atts

    def get_attributes_by_att_types(self, att_types: list):
        filtered = []
        for att in self.attribute_to_attribute_type.keys():
            if self.attribute_to_attribute_type[att] in att_types:
                filtered.append(att)
        return filtered

    def _initialize_maps(self):
        if self.timestamp is not None:
            self.cases = self.df.sort_values([self.case_id, self.timestamp],
                                             ascending=[True, True]).groupby(self.case_id)

        else:
            self.cases = self.augmented_df.sort_values([self.case_id], ascending=[True]).groupby(
                self.case_id)
        for att in self.df.columns:
            if "Unnamed:" in att:
                continue
            if len(self.df[att]) > 50:
                self.samples[att] = self.df[att].sample(n=50).values
            else:
                self.samples[att] = pd.Series()
            self.cleaned_labels[att] = list()
            self.attribute_to_attribute_type[att] = AttributeType.UNKNOWN
            self.attribute_to_concept_type[att] = ConceptType.OTHER.value
            if "case:" in att:
                self.case_attributes.append(att)
            elif all(len(current_case[att].dropna().unique()) == 1 for case_id, current_case in self.cases):
                self.case_attributes.append(att)


    def get_df_representation_filtered(self, att_type: AttributeType, copy=False):
        filtered = []
        for d in self.get_all_duplicates():
            del self.attribute_to_concept_type[d]
            del self.attribute_to_attribute_type[d]
        for att in self.attribute_to_attribute_type.keys():
            if self.attribute_to_attribute_type[att] == att_type:
                filtered.append(att)
        if copy:
            return self.df[filtered].copy()
        return self.df[filtered]

    def get_label_attribute(self):
        for att in self.attribute_to_attribute_type.keys():
            if self.attribute_to_attribute_type[att] == AttributeType.LABEL:
                return att
        return None

    def set_cleaned_df(self, df: pd.DataFrame):
        self.cleaned_df = df

    def get_cleaned_df(self):
        return self.cleaned_df

    def get_nes_for_label(self, label):
        """
        :param label: the label to check
        :return: a list of named entities in the label if any (format :[(ne, start, end, type)])
        """
        if label not in self.label_to_ne.keys():
            return []
        return self.label_to_ne[label]

    def get_all_unique_values_for_role(self, role):
        return set(list(itertools.chain.from_iterable([find_role(self.tagged_labels[x][0], self.tagged_labels[x][1], role) for x in self.tagged_labels.keys()])))

    def to_separate_result(self):
        """
        Combines the semantic information extracted from the initial even log in a new augmented event log
        :return: the resulting data frame with dedicated attribute per semantic role instance, i.e.
        separate attributes for each business object for instance.
        """
        res_log = self.df

        rich_text_attributes = [att for att in self.get_attributes_by_att_types(consider_for_tagging) if
                                self.attribute_to_concept_type[att] == ConceptType.OTHER.value]

        print("Rich", rich_text_attributes)

        other_attributes = self.get_attributes_by_att_types(consider_for_label_classification) + [att for att in self.get_attributes_by_att_types(consider_for_tagging) if att not in rich_text_attributes]

        print("Other", other_attributes)
        flag_attributes = self.get_attributes_by_att_types([AttributeType.FLAG])
        cnt = type_mapping.copy()
        for key in cnt.keys():
            cnt[key] = 0
        for key in type_mapping.keys():
            s = (
                self.cleaned_df.apply(
                    lambda x: ",".join(list(itertools.chain.from_iterable(
                        [find_role(self.tagged_labels[x[att]][0], self.tagged_labels[x[att]][1], key) for att in
                         rich_text_attributes if
                         not x[att] in TERMS_FOR_MISSING and x[att] in self.tagged_labels.keys()]))), axis=1).str.split(
                    ",", expand=True)
                    .stack()
                    .to_frame("words")
                    .reset_index(1, drop=True)
            )
            s["count"] = s.groupby(level=0).cumcount()
            new_part = s.rename_axis("idx").groupby(["idx", "count"])["words"].agg(" ".join).unstack(1)
            new_part.columns = [type_mapping[key] + ':' + str(col) for col in new_part.columns if col != 'idx']
            res_log = pd.concat([res_log, new_part], axis=1).reindex(res_log.index)
            for att in other_attributes:

                if key == ConceptType.BO_PROP.value:
                    if type_mapping[key] == self.attribute_to_concept_type[att]:
                        res_log[type_mapping[key] + ':a:' + str(cnt[key]) + att] = res_log[att]
                        cnt[key] += 1
                else:
                    if type_mapping[key] == self.attribute_to_concept_type[att]:
                        res_log[type_mapping[key] + ':a:' + str(cnt[key])] = res_log[att]
                        cnt[key] += 1

            if len(flag_attributes) > 0:
                res_log[type_mapping[key] + ':a:' + str(cnt[key])] = ""
            for att in flag_attributes:
                if type_mapping[key] == self.attribute_to_concept_type[att] == type_mapping[
                ConceptType.BO_STATUS.value]:
                    res_log[type_mapping[key] + ':a:' + str(cnt[key])] = res_log[type_mapping[key] + ':a:' + str(cnt[key])] + res_log[
                        att].apply(lambda
                                       x: att+',' if x == 'true' or x == 1 or x == 'TRUE' else '').astype(str)
                elif type_mapping[key] == self.attribute_to_concept_type[att]:
                    res_log[type_mapping[key] + ':a:' + str(cnt[key])] = res_log[att].astype(str)
                    cnt[key] += 1
        return res_log

    def to_result_log_full(self, expanded=True, add_refined_label=False):
        """
        Combines the semantic information extracted from the initial even log in a new enhanced event log
        :return: the resulting data frame with dedicated attribute per semantic component type
        """
        self.map_tags()
        print(self.attribute_to_concept_type)
        if expanded:
            res_log = self.to_separate_result()
        else:
            flag_attributes = self.get_attributes_by_att_types([AttributeType.FLAG])
            rich_text_attributes = [att for att in self.get_attributes_by_att_types(consider_for_tagging) if self.attribute_to_concept_type[att] == ConceptType.OTHER.value]
            other_attributes = self.get_attributes_by_att_types(consider_for_label_classification) + [att for att in self.get_attributes_by_att_types(consider_for_tagging) if self.attribute_to_concept_type[att] != ConceptType.OTHER.value]
            res_log = self.df
            for key in type_mapping.keys():
                if key == ConceptType.BO_PROP.value:
                    # It doesn't make sense to copy all different properties into another attribute,
                    # as the mapping to the attribute names are lost
                    continue
                res_log[type_mapping[key]] = self.cleaned_df.apply(
                    lambda x: ",".join(list(itertools.chain.from_iterable(
                        [find_role(self.tagged_labels[x[att]][0], self.tagged_labels[x[att]][1], key) for att in
                         rich_text_attributes if
                         not x[att] in TERMS_FOR_MISSING and x[att] in self.tagged_labels.keys()]))), axis=1)

                res_log[type_mapping[key] + ':a'] = ""
                for att in other_attributes:
                    if type_mapping[key] == self.attribute_to_concept_type[att]:
                        res_log[type_mapping[key] + ':a'] = res_log[type_mapping[key] + ':a'] + ", " + res_log[att].astype(str)
                if len(flag_attributes) > 0:
                    res_log[type_mapping[key] + ':a'] = ""
                for att in flag_attributes:
                    if type_mapping[key] == self.attribute_to_concept_type[att] == type_mapping[
                        ConceptType.BO_STATUS.value]:
                        res_log[type_mapping[key] + ':a'] = res_log[type_mapping[key] + ':a'] + res_log[
                            att].apply(lambda
                                           x: att+',' if x == 'true' or x == 1 or x == 'TRUE' else '').astype(str)
                    elif type_mapping[key] == self.attribute_to_concept_type[att]:
                        res_log[type_mapping[key] + ':a'] = res_log[type_mapping[key] + ':a'] + " " + res_log[att]
        if add_refined_label:
            res_log["old:"+self.event_label] = res_log[self.event_label]
            res_log[self.event_label] = self.cleaned_df.apply(lambda x: list(itertools.chain.from_iterable(
                [find_components(self.tagged_labels[x[att]][0], self.tagged_labels[x[att]][1]) for att in
                 rich_text_attributes if not x[att] in TERMS_FOR_MISSING and x[att] in self.tagged_labels.keys()])),
                                                            axis=1)
            res_log['concept:name'] = res_log.apply(
                lambda x: "".join(x['concept:name']) if len(x['concept:name']) == 1 else x['concept:name'], axis=1)
        res_log = res_log.replace("", np.nan).dropna(axis=1, how='all')
        res_log[UNIQUE_EVENT_ID] = res_log.index + 1
        self.augmented_df = res_log
        return res_log

    def get_all_duplicates(self):
        res = set()
        for d in self.duplicates:
            for v in self.duplicates[d]:
                res.add(v)
        return res

    def map_tags(self):
        for k, v in self.attribute_to_concept_type.items():
            if v == 'X' or v =='IGNORE':
                continue
            self.attribute_to_concept_type[k] = type_mapping[v]
