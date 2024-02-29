import readwrite.loader as load
from preprocessing.preprocessor import clean_attribute_name
from data.gathering.schemaorgextraction import read_and_extract
from const import *


def transform_to_training_data(attribute_data):
    docs = []
    labels = []
    for origin in attribute_data.keys():
        for role in attribute_data[origin].keys():
            for att_name in attribute_data[origin][role]:
                if att_name not in docs:
                    docs.append(clean_attribute_name(att_name))
                    labels.append(role)
    return docs, labels


class LabeledData:
    def __init__(self, config, label_file='ACTIVITIES.txt', attribute_file='attributes_labeled.json'):
        self.config = config
        self.vocab = {}
        self._load_data(config.resource_dir, label_file, attribute_file)

    def get_label_data(self):
        return self.vocab[INSTANCE_LEVEL_DATA]

    def get_attribute_data(self):
        if len(self.config.exclude_data_origin) > 0:
            filtered_data = {}
            for key in self.vocab[ATTRIBUTE_LEVEL_DATA].keys():
                if key not in self.config.exclude_data_origin:
                    filtered_data[key] = self.vocab[ATTRIBUTE_LEVEL_DATA][key]
            return transform_to_training_data(filtered_data)
        return transform_to_training_data(self.vocab[ATTRIBUTE_LEVEL_DATA])

    def _load_data(self, directory, label_file, log_file):
        self.vocab[INSTANCE_LEVEL_DATA] = load.get_tagged_labels(directory, label_file)
        self.vocab[ATTRIBUTE_LEVEL_DATA] = load.get_annotated_attributes(directory, log_file)
        actor_terms, act_terms, action_status_terms, obj_terms, obj_status_terms, obj_property_terms = read_and_extract(
            self.config.resource_dir)
        self.vocab[ATTRIBUTE_LEVEL_DATA][SCHEMA_ORG] = {ConceptType.BO_NAME.value: obj_terms,
                                                        ConceptType.ACTOR_NAME.value: actor_terms,
                                                        ConceptType.ACTION_NAME.value: act_terms,
                                                        ConceptType.BO_PROP.value: obj_property_terms,
                                                        ConceptType.BO_STATUS.value: obj_status_terms}
        # self.vocab[ATTRIBUTE_LEVEL_DATA][BPMAI] = {
        #     ConceptType.ACTOR_NAME.value: list(extract_resources_from_jsons(self.config))}

    def __repr__(self):
        return str(self.vocab)
