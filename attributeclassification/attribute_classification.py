from preprocessing.nlp_utils import TERMS_FOR_MISSING
import preprocessing.preprocessor as pp
import time

import random
from const import AttributeType, consider_for_label_classification, consider_for_value_classification, \
    consider_for_tagging, const_to_tag, ConceptType, EVAL_MODE, COMMON_ABBREV, SCHEMA_ORG_OBJ_PROP
from data.labeled_data import LabeledData
from attributeclassification.subclassifiers.att_label_classifier import AttributeLabelClassifier
from preprocessing.preprocessor import clean_attribute_name
from data.gathering.data_generator import DataGenerator, get_plain_sentence
from model.augmented_log import AugmentedLog
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer





class AttributeClassifier:

    def __init__(self, config, word_embeddings):
        self.config = config
        self.data = LabeledData(self.config)
        self.word_embeddings = word_embeddings
        self.lemmatizer = WordNetLemmatizer()

    def check_proper_object(self, tok):
        boolean = self.lemmatizer.lemmatize(tok) in words.words() and tok not in COMMON_ABBREV and tok not in SCHEMA_ORG_OBJ_PROP
        #print(boolean, tok)
        return boolean

    def find_objects(self, split, tags):
        res = [tok for tok, bo in zip(split, tags) if
               (bo == 'BO' or (bo == "X" and tok in stopwords.words('english'))) and self.check_proper_object(tok)]
        if len(res) > 0 and res[0] in stopwords.words('english'):
            res = res[1:]
        if len(res) > 0 and res[-1] in stopwords.words('english'):
            res = res[0:-1]
        return res

    def get_role_by_semsim_classification(self, aug_log: AugmentedLog):
        return AttributeLabelClassifier(self.config, self.data, aug_log).with_tf_idf_and_embedding(
            self.word_embeddings.embeddings, eval_mode=self.config.mode is EVAL_MODE)

    def get_resource_type(self, aug_log: AugmentedLog):
        return ResourceClassifier(self.config, self.data, aug_log).classify()

    def get_default_attribute_classes(self, att_name):
        """
        Looks for match in lookup table and assigns corresponding concept
        If none present in the lookup table tries to annotate using a .model
        :param ld:
        """
        if att_name in const_to_tag.keys():
            return const_to_tag[att_name]
        else:
            return None

    def classify_masked(self, sen_idx, entity_type, unique_val, classifier):
        unique_val = str(unique_val)
        idx = []
        new_sen = sen_idx[0].copy()
        if sen_idx[1] >= 0:
            for k in range(len(unique_val.split())):
                idx.append((sen_idx[1] + k))
                new_sen.insert((sen_idx[1] + k), (unique_val.split()[k], entity_type))
            return classifier.predict_single_label_full(get_plain_sentence(new_sen)), idx
        return None

    def find_state_pattern_in_label_using_bert(self, ld: AugmentedLog, classifier):
        att_to_state = dict()
        for att in ld.get_attributes_by_att_types(consider_for_label_classification):
            split, tags = classifier.predict_single_label(clean_attribute_name(att))
            if len(tags) == 1 and (tags[0] == ConceptType.BO_STATUS.value):
                att_to_state[att] = ConceptType.BO_STATUS.value
            elif len(tags) == 2 and (((tags[0] == ConceptType.ACTION_NAME.value or tags[
                0] == ConceptType.BO_STATUS.value) and tags[1] == ConceptType.OTHER.value) or ((tags[
                                                                                                    1] == ConceptType.ACTION_NAME.value or
                                                                                                tags[
                                                                                                    1] == ConceptType.BO_STATUS.value) and
                                                                                               tags[
                                                                                                   0] == ConceptType.OTHER.value)):
                att_to_state[att] = ConceptType.BO_STATUS.value

        return att_to_state

    def make_sen_for_type(self, sen, entity_type):
        local_sen = sen.copy()
        remove_idx = []
        i = 0
        for item in local_sen:
            if item[1] == entity_type:
                remove_idx.append(i)
            i += 1
        masked = local_sen.copy()
        for idx in remove_idx:
            masked.remove(local_sen[idx])
        res_idx = -1
        if len(remove_idx) > 0:
            res_idx = remove_idx.pop(0)
        return masked, res_idx

    def build_masked_sentences(self, sen):
        entity_type_to_sen = {ConceptType.ACTOR_NAME.value: '', ConceptType.ACTION_NAME.value: '',
                              ConceptType.PASSIVE_NAME.value: '',
                              ConceptType.BO_NAME.value: '', ConceptType.OTHER.value: '',
                              ConceptType.ACTION_STATUS.value: '', ConceptType.BO_STATUS.value: ''}
        for entity_type in entity_type_to_sen.keys():
            entity_type_to_sen[entity_type] = self.make_sen_for_type(sen, entity_type)
        return entity_type_to_sen

    def determine_att_class_by_masking(self, sen_gen: DataGenerator, ld: AugmentedLog, classifier, sample_size=10):
        """
        determine potential entity type for column by feeding the unique column values into BERT
        masking each entity type for each unique value once and inserting that value instead.
        - ignores values that are only numeric, a date or contain only one character
        """
        the_dict = {}
        sens = sen_gen.get_defined_sentences()
        winners = {}
        for att in ld.get_attributes_by_att_types(consider_for_value_classification):
            winners[att] = []
        for sen in sens:
            masked_sentences = self.build_masked_sentences(sen)
            col_annotations = {}
            for att in ld.get_attributes_by_att_types([AttributeType.TEXT]):
                col_annotations[att] = 'X'
                unique_values = ld.att_to_unique[att]
                if len(unique_values) > sample_size:
                    unique_values = random.sample(unique_values, sample_size)
                aggregated_scores = {ConceptType.ACTOR_NAME.value: [0], ConceptType.ACTION_NAME.value: [0],
                                     ConceptType.PASSIVE_NAME.value: [0], ConceptType.BO_NAME.value: [0],
                                     ConceptType.OTHER.value: [0], ConceptType.ACTION_STATUS.value: [0],
                                     ConceptType.BO_STATUS.value: [0]}
                for unique_val in unique_values:
                    the_dict[unique_val] = {}
                    for entity_type in masked_sentences.keys():
                        curr = self.classify_masked(masked_sentences[entity_type], entity_type, unique_val, classifier)
                        if curr is not None:
                            the_dict[unique_val][entity_type] = curr[0]
                            idx = curr[1]
                            split_res = curr[0]
                            res = split_res[1]
                            tags = res[0][0]
                            prob = res[1][0]
                            pred = []
                            act = []
                            num_matches = 0
                            for i in idx:
                                if tags[i] == entity_type:
                                    num_matches += 1
                                pred.append(tags[i])
                                act.append(entity_type)
                            # all should be correct
                            if num_matches >= len(act):
                                aggregated_scores[entity_type].append(1)
                winner = 'X'
                curr_max = 0.0
                for entity_type in aggregated_scores.keys():
                    curr_scor = sum(aggregated_scores[entity_type]) / len(aggregated_scores[entity_type])
                    if curr_scor > curr_max:
                        curr_max = curr_scor
                        winner = entity_type
                col_annotations[att] = (winner, curr_max)
                winners[att].append(winner)
        for att in winners.keys():
            winners[att] = max(set(winners[att]), key=winners[att].count)
        return winners

    def handle_exclusive_type_attributes(self, ld: AugmentedLog, bo_ratio=0.5):
        for att in ld.get_attributes_by_att_types(consider_for_tagging):
            unique_vals = ld.att_to_unique[att]

            if sum([all(comp == ConceptType.BO_NAME.value or comp == ConceptType.OTHER.value for tok, comp in
                        zip(ld.tagged_labels[unique_val][0], ld.tagged_labels[unique_val][1])) for unique_val in
                    unique_vals]) / len(unique_vals) >= bo_ratio:
                #print(att + " is reassigned")
                ld.set_attribute_type(att, AttributeType.TEXT)
            elif sum([all(comp == ConceptType.ACTION_STATUS.value for tok, comp in
                          zip(ld.tagged_labels[unique_val][0], ld.tagged_labels[unique_val][1])) for unique_val in
                      unique_vals]) / len(unique_vals) >= bo_ratio:
                #print(att + " is predominantly action status")
                ld.set_concept_type(att, ConceptType.ACTION_STATUS)
            elif sum([all(comp == ConceptType.BO_STATUS.value for tok, comp in
                          zip(ld.tagged_labels[unique_val][0], ld.tagged_labels[unique_val][1])) for unique_val in
                      unique_vals]) / len(unique_vals) >= bo_ratio:
                #print(att + " is predominantly object status")
                ld.set_concept_type(att, ConceptType.BO_STATUS.value)
            elif sum([all(comp == ConceptType.ACTOR_NAME.value for tok, comp in
                          zip(ld.tagged_labels[unique_val][0], ld.tagged_labels[unique_val][1])) for unique_val in
                      unique_vals]) / len(unique_vals) >= bo_ratio:
                #print(att + " is predominantly actor info")
                ld.set_concept_type(att, ConceptType.ACTOR_NAME.value)
            elif sum([all(comp == ConceptType.ACTOR_NAME.value for tok, comp in
                          zip(ld.tagged_labels[unique_val][0], ld.tagged_labels[unique_val][1])) for unique_val in
                      unique_vals]) / len(unique_vals) >= bo_ratio:
                #print(att + " is predominantly action info")
                ld.set_concept_type(att, ConceptType.ACTION_NAME.value)

    def check_instance_basic(self, real_samples):
        res = (sum(
            any(char.isdigit() for char in str(sample)) or pp.check_for_uuid(str(sample)) for sample in
            real_samples) / len(
            real_samples) > self.config.instance_thresh) or (all(len(str(sample)) == 1 for sample in real_samples))
        return res

    def base_disambiguation(self, ld: AugmentedLog):
        copy = ld.attribute_to_concept_type.copy()
        for k, v in copy.items():
            if ld.attribute_to_concept_type[k] == ConceptType.BO_INSTANCE.value:
                continue
            real_samples = [sample for sample in ld.samples[k] if str(sample) not in TERMS_FOR_MISSING]
            if len(real_samples) == 0:
                ld.set_concept_type(k, ConceptType.OTHER)
                continue
            if v == ConceptType.ACTOR_NAME.value and (self.check_instance_basic(real_samples) or self.check_ne(ld, k)):
                ld.set_concept_type(k, ConceptType.ACTOR_INSTANCE)
            if v == ConceptType.ACTION_NAME.value and self.check_instance_basic(real_samples):
                ld.set_concept_type(k, ConceptType.ACTION_INSTANCE)
            if v == ConceptType.PASSIVE_NAME.value and self.check_instance_basic(real_samples):
                ld.set_concept_type(k, ConceptType.PASSIVE_INSTANCE)
            if v == ConceptType.BO_NAME.value and self.check_instance_basic(real_samples):
                ld.set_concept_type(k, ConceptType.BO_INSTANCE)

    def check_ne(self, ld: AugmentedLog, att):
        if ld.attribute_to_attribute_type[att] not in consider_for_value_classification and \
                ld.attribute_to_attribute_type[
                    att] != AttributeType.STRING:
            return False
        if att in ld.attribute_to_ne.keys():
            return True
        num_unique = len(ld.att_to_unique[att])
        count_gpe = 0.0
        for label in ld.att_to_unique[att]:
            nes = ld.get_nes_for_label(label)
            if len(nes) > 0:
                for entry in nes:
                    if entry[3] == 'PERSON' or entry[3] == 'ORG' or entry[3] == 'GPE' or entry[3] == 'LAW' or entry[3] == 'PRODUCT' or entry[3] == 'WORK_OF_ART':
                        count_gpe += 1.0
        if count_gpe / num_unique > self.config.instance_thresh:
            return True
        else:
            return False

    def get_bert_annotations(self, ld, bert_tagger):
        res = {}
        # generate a random sentence for column concept detection
        sen_gen = self.data.get_label_data()
        tic = time.perf_counter()
        bert_annotations = self.determine_att_class_by_masking(sen_gen, ld, bert_tagger)
        toc = time.perf_counter()
        #print(f"Computed concept by masking in {toc - tic:0.4f} seconds")
        for curr in bert_annotations.keys():
            if bert_annotations[curr] == ConceptType.PASSIVE_NAME.value or bert_annotations[
                curr] == ConceptType.ACTOR_NAME.value:
                res[curr] = ConceptType.ACTOR_NAME.value
            elif bert_annotations[curr] == ConceptType.BO_NAME.value:
                res[curr] = ConceptType.BO_NAME.value
            else:
                res[curr] = ConceptType.OTHER.value
        return res

    def run(self, aug_log: AugmentedLog, bert_tagger):
        """
            :param aug_log: the log (log including annotations and tagged labels)
            :param bert_tagger: the BERT .model to be used
            :return:
            """
        if self.config.resource_type:
            self.get_resource_type(aug_log=aug_log)
        min_unique_action = 3
        # Detect BO or ACTOR from noun only
        self.handle_exclusive_type_attributes(aug_log)
        bert_annotations = self.get_bert_annotations(aug_log, bert_tagger)
        state_labels = self.find_state_pattern_in_label_using_bert(aug_log, bert_tagger).keys()
        tic = time.perf_counter()
        kb_annotations = self.get_role_by_semsim_classification(aug_log)
        toc = time.perf_counter()
        #print(f"Computed roles based on semantic similarity in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()
        # Apply some heuristics to eliminate obvious errors
        # 1. attribute names that are recognized as object:status are considered as such
        # 2. if only the similarity based approach was used, use this result
        # 3. if both approaches have results:
        #   3.1 if they disagree, check if the similarity score is above threshold, if yes pick that result
        #   3.2 if they disagree, check if the similarity score is below min threshold, if yes pick BERT result
        for col in aug_log.get_attributes_by_att_types(consider_for_label_classification):
            if self.get_default_attribute_classes(col) is not None:
                default_att = self.get_default_attribute_classes(col)
                if default_att is not None:
                    aug_log.set_concept_type(col, default_att)
            elif col in state_labels:
                aug_log.set_concept_type(col, ConceptType.BO_STATUS.value)
            elif aug_log.get_concept_annotation()[col] == ConceptType.OTHER.value and col in kb_annotations.keys():
                if col in bert_annotations.keys():
                    if bert_annotations[col] == ConceptType.PASSIVE_NAME.value:
                        aug_log.set_concept_type(col, ConceptType.ACTOR_NAME.value)
                    else:
                        aug_log.set_concept_type(col, bert_annotations[col])
                else:
                    if kb_annotations[col][0] == ConceptType.PASSIVE_NAME.value or kb_annotations[col][
                        0] == ConceptType.ACTOR_NAME.value:
                        aug_log.set_concept_type(col, ConceptType.ACTOR_NAME.value)
                    elif kb_annotations[col][0] == ConceptType.ACTION_NAME.value and aug_log.num_uniques[
                        col] >= min_unique_action and aug_log.attribute_to_attribute_type[col] != AttributeType.FLAG:
                        aug_log.set_concept_type(col, ConceptType.ACTION_NAME.value)
                    elif kb_annotations[col][0] == ConceptType.BO_NAME.value and aug_log.num_uniques[col] >= 1 and \
                            aug_log.attribute_to_attribute_type[col] != AttributeType.FLAG:
                        aug_log.set_concept_type(col, ConceptType.BO_NAME.value)
                    elif kb_annotations[col][0] == ConceptType.ACTION_STATUS.value:
                        aug_log.set_concept_type(col, ConceptType.ACTION_STATUS.value)
                    elif kb_annotations[col][0] == ConceptType.BO_STATUS.value:
                        aug_log.set_concept_type(col, ConceptType.BO_STATUS.value)
                    elif kb_annotations[col][0] == ConceptType.BO_PROP.value:
                        aug_log.set_concept_type(col, ConceptType.BO_PROP.value)
                    else:
                        aug_log.set_concept_type(col, ConceptType.OTHER.value)
        # disambiguate instance and type level

        for att in aug_log.attribute_to_concept_type.keys():

            if aug_log.attribute_to_concept_type[att] in \
                    [ConceptType.BO_NAME.value, ConceptType.BO_PROP.value, ConceptType.ACTOR_NAME.value, ConceptType.ACTOR_INSTANCE.value] \
                     and att != "case:concept:name" and aug_log.attribute_to_attribute_type[att] not in [AttributeType.FLAG, AttributeType.TIMESTAMP, AttributeType.NUMERIC]: #and att in aug_log.case_attributes
                preprocessed_att = clean_attribute_name(att)
                _, tags = bert_tagger.predict_single_label(preprocessed_att)
                #print(preprocessed_att, tags)
                if any(tag == ConceptType.BO_NAME.value for tag in tags):
                    obj_type = self.lemmatizer.lemmatize(" ".join(self.find_objects(preprocessed_att.split(" "), tags)))
                    if obj_type != "":
                        #print(att, obj_type)
                        aug_log.case_level_bo.add(obj_type)
                        aug_log.case_bo_to_case_att[obj_type] = att

        self.base_disambiguation(aug_log)
        for att in aug_log.attribute_to_concept_type.keys():
            if aug_log.attribute_to_concept_type[att] == ConceptType.BO_NAME.value:
                aug_log.attribute_to_concept_type[att] = ConceptType.OTHER.value
        #print(aug_log.attribute_to_concept_type)
        # The function call below maps the internal naming of the tags to the naming in the paper
        toc = time.perf_counter()

        print(f"Labeled attributes in {toc - tic:0.4f} seconds")
        #return kb_annotations, bert_annotations, bert_annotations, aug_log.attribute_to_concept_type
        return {att: (role, 1) for att, role in aug_log.attribute_to_concept_type.items()}

