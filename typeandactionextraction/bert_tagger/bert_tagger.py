import pandas as pd
import random
from model.augmented_log import AugmentedLog
from const import consider_for_tagging, ConceptType
from preprocessing.nlp_utils import check_propn

from typeandactionextraction.bert_tagger import BertWrapper, BertForLabelParsing


def get_train_val_test_split(sentences):
    random.shuffle(sentences)

    train_pct_index = int(0.8 * len(sentences))
    train_data = sentences[:train_pct_index]
    test_data = sentences[train_pct_index:]

    val_pct_index = int(0.8 * len(train_data))
    val_data = train_data[val_pct_index:]
    train_data = train_data[:val_pct_index]
    return train_data, val_data, test_data


def check_tok_for_object_type(split, pred):
    new_pred = []
    i = 1
    for tok, lab in zip(split, pred):
        if len(split) > i and tok == ConceptType.BO_NAME.value and split[i] != ConceptType.BO_NAME.value and check_propn(tok):
            new_pred.append(ConceptType.OTHER.value)
        else:
            new_pred.append(lab)
        i += 1
    return new_pred

class BertTagger:

    def __init__(self, config):
        self.config = config
        self.model = None
        self._load_trained_model(config.model_dir)

    def _load_trained_model(self, path):
        self.model = BertWrapper.load_serialized(path, BertForLabelParsing)

    def tag_log(self, ld: AugmentedLog):
        return self._add_tags(ld)

    def _add_tags(self, ld: AugmentedLog):
        res = pd.DataFrame()
        seen_tagged = ld.tagged_labels
        cols_to_be_tagged = ld.get_attributes_by_att_types(consider_for_tagging)
        for v in cols_to_be_tagged:
            unique_labels = ld.get_cleaned_df()[v].unique()
            for unique in unique_labels:
                unique = str(unique)
                if unique not in seen_tagged:
                    seen_tagged[unique] = self.predict_single_label(unique)
        for v in cols_to_be_tagged:
            res[v] = ld.get_cleaned_df()[v].apply(lambda x: BertTagger._fill_all(x, seen_tagged))
        return res

    def get_tags_for_df(self, ld: AugmentedLog) -> dict:
        seen_tagged = ld.tagged_labels
        cols_to_be_tagged = ld.get_attributes_by_att_types(consider_for_tagging)
        for v in cols_to_be_tagged:
            unique_labels = ld.att_to_unique[v]
            predicted = self.predict_batch_at_once(unique_labels)
            for unique, pred in zip(unique_labels, predicted):
                if unique not in seen_tagged:
                    pred = check_tok_for_object_type(unique.split(), pred)
                    seen_tagged[unique] = unique.split(), pred
        ld.add_tagged_vals(seen_tagged)
        return seen_tagged

    def get_tags_for_col(self, ld: AugmentedLog, col_name) -> dict:
        seen_tagged = {}
        print('semantic tagging: ' + col_name)
        unique_labels = ld.att_to_unique[col_name]
        predicted = self.predict_batch_at_once(unique_labels)
        for unique, pred in zip(unique_labels, predicted):
            if unique not in seen_tagged:
                pred = check_tok_for_object_type(unique.split(), pred)
                seen_tagged[unique] = unique.split(), pred
        return seen_tagged

    def get_tags_for_list(self, li: list) -> dict:
        tagged = {}
        for unique in li:
            unique = str(unique)
            if unique not in tagged:
                tagged[unique] = self.predict_single_label(unique)[1]
        return tagged

    def predict_single_label(self, label):
        split, pred = label.split(), self.model.predict([label.split()])[0][0]
        pred = check_tok_for_object_type(split, pred)
        return split, pred

    def predict_batch_at_once(self, labels):
        return self.model.predict([label.split() for label in labels])[0]

    def predict_single_label_full(self, label):
        return label.split(), self.model.predict([label.split()])

    @staticmethod
    def _fill_all(x, seen_tagged):
        uniquely_tagged = []
        tagging = str()
        if x not in seen_tagged.keys():
            return
        for i in range(len(seen_tagged[x][0])):
            tagging = tagging + str(seen_tagged[x][0][i]) + '<>' + str(seen_tagged[x][1][i]) + ', '
        uniquely_tagged.append(tagging)
        return uniquely_tagged

    def serialize_model(self):
        self.model.save_serialize('./.model/')


