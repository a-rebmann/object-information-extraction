import json
import os
import sys
import time

from instancediscovery.instancediscoverer import InstanceDiscoverer
from model.log.log import from_pd_log
from propertytoinstance.propertyassignment import PropertyAssigner
from instancetoevent.instanceassignment import InstanceAssigner
from readwrite import writer, loader
import downstream.transformation.transform as transform
from data.word_embeddings import WordEmbeddings
from attributeclassification.attribute_classification import AttributeClassifier
from model.augmented_log import AugmentedLog
import preprocessing.preprocessor as pp
from typeandactionextraction.bert_tagger.bert_tagger import BertTagger
from config import Config
from const import MatchingMode, MAIN_MODE, ConceptType, type_mapping, EVAL_MODE

# DIRECTORIES
# default input directory
DEFAULT_INPUT_DIR = 'input/logs/'
# default output directory
DEFAULT_OUTPUT_DIR = 'output/logs/'
# default directory where the models are stored
DEFAULT_MODEL_DIR = '.model/main/'
# default directory for resources
DEFAULT_RES_DIR = 'resources/'

DEFAULT_CONFIG = Config(input_dir=DEFAULT_INPUT_DIR, model_dir=DEFAULT_MODEL_DIR, resource_dir=DEFAULT_RES_DIR,
                        output_dir=DEFAULT_OUTPUT_DIR, bo_ratio=0.5, conf_thresh=0.5, instance_thresh=0.5,
                        exclude_data_origin=[], we_file="glove-wiki-gigaword-50", matching_mode=MatchingMode.SEM_SIM,
                        mode=MAIN_MODE, res_type=True)


def augment_and_transform_log(directory, name, config):
    print(name)
    timing = {}
    aug_log = loader.deserialize_event_log(config.resource_dir, name)
    # STEP 0 loading the log and doing semantic component extraction
    if aug_log is False:
        tic = time.perf_counter()
        df, case_id, event_label, time_stamp = loader.load_log(directory, name)
        aug_log = AugmentedLog(name, df, case_id, event_label=event_label, timestamp=time_stamp)
        if all(len(case) <= 1 for cid, case in aug_log.cases):
            print("ABORTING as all cases have at most one event")
        toc = time.perf_counter()
        print(f"Loaded the current log in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()
        pp.pre_process(config, aug_log)
        toc = time.perf_counter()
        timing[0] = toc - tic
        print(f"Preprocessed the current log in {toc - tic:0.4f} seconds")
        writer.serialize_event_log(config.resource_dir, aug_log)
    print("load word embeddings " + config.word_embeddings_file)
    # STEP 1 Extracting Object types and actions
    print(f"Step 1: extracting type and action info")
    tic_1 = time.perf_counter()
    word_embeddings = WordEmbeddings(config=config)
    if not aug_log.augmented_df:
        print("BERT-based semantic tagging")
        print("obtain .model")
        tic = time.perf_counter()
        bert_tagger = BertTagger(config=config)
        toc = time.perf_counter()
        print(f"Loaded the trained .model in {toc - tic:0.4f} seconds")
        print('semantic tagging text attributes')
        bert_tagger.get_tags_for_df(aug_log)
        # for key, val in aug_log.tagged_labels.items():
        #     print(key, val)
        tic = time.perf_counter()
        print(f"Tagged the whole data set in {tic - toc:0.4f} seconds")
        tic = time.perf_counter()
        print('starting attribute classification')
        attribute_classifier = AttributeClassifier(config=config, word_embeddings=word_embeddings)
        attribute_classifier.run(aug_log=aug_log, bert_tagger=bert_tagger)
        toc = time.perf_counter()
        print(f"Attribute classification finished within {toc - tic:0.4f} seconds")
        #aug_log.to_result_log_full(expanded=True, add_refined_label=False)
        #writer.serialize_event_log(config.resource_dir, aug_log)
    aug_log.map_tags()
    print(aug_log.case_bo_to_case_att, "Case-level object")
    for col in aug_log.cleaned_df.columns:
        aug_log.df[col+"_cleaned"] = aug_log.cleaned_df[col]

    #writer.create_file_for_df(aug_log.augmented_df, config.output_dir, aug_log.name)
    if check_multiple_roles(aug_log, ConceptType.BO_NAME.value):
        print("Multiple object types were found -> object-centric event log can be created")
    print(f"Transforming log adding extracted object type and action info")

    #log = False#loader.deserialize_new_log(DEFAULT_RES_DIR, name)
    #if not log:
    tic = time.perf_counter()
    log = from_pd_log(aug_log, aug_log.name, config)
    toc = time.perf_counter()
    print(f"Log transformation finished within {toc - tic:0.4f} seconds")
    #writer.serialize_new_log(config.resource_dir, log)
    print("Log has ", len(log.cases), "cases and", len(log.events), "events.")
    toc_1 = time.perf_counter()
    print(f"Step 1: Extraction finished within {toc_1 - tic_1:0.4f} seconds")
    # STEP 2 Discovering object instances
    print(f"Step 2: Starting instance discovery.")
    tic = time.perf_counter()
    instance_discoverer = InstanceDiscoverer(config)
    instance_discoverer.discover_instances(log, aug_log.case_bo_to_case_att)
    toc = time.perf_counter()
    timing[2] = toc - tic
    print(f"Step 2: Instance discovery finished within {toc - tic:0.4f} seconds")
    # # STEP 3 Assign properties to instances of objects
    print(f"Step 3: Starting property assignment.")
    tic = time.perf_counter()
    property_assigner = PropertyAssigner(config=config, log=log, instance_disc=instance_discoverer)
    property_assigner.assign_properties(log=log, candidates=instance_discoverer.property_candidates, instance_attributes=instance_discoverer.object_to_id)
    toc = time.perf_counter()
    timing[3] = toc - tic
    print(f"Step 3: Property assignment finished within {toc - tic:0.4f} seconds")
    # STEP 4 log-level object instance identification and event allocation
    print(f"Step 4: Starting instance-to-event assignment.")
    tic = time.perf_counter()
    instance_assigner = InstanceAssigner(config=config, log=log, instance_discoverer=instance_discoverer)
    instance_assigner.assign_instances_to_events(object_to_property=property_assigner.object_to_property)

    for obj_type, atts in property_assigner.object_to_property.items():
        for event in log.events:
            for att in atts:
                if att in event.vmap:
                    for inst in event.omap:
                        log.objects[inst].vmap[att] = event.vmap[att]
                    del event.vmap[att]
    toc = time.perf_counter()
    timing[4] = toc - tic
    print(f"Step 4: Instance-to-event assignment finished within {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    oclog = transform.create_ocel_from_log(log=log)
    toc = time.perf_counter()
    timing[5] = toc - tic
    print(f"Create output: OCEL construction took {toc - tic:0.4f} seconds")
    # Directly from dictionary
    if config.mode == EVAL_MODE:
        ocel_name = writer.validate_and_write_ocel(oclog, config.output_dir, aug_log.name, config.resource_dir, config)
        return ocel_name, log, aug_log, instance_discoverer, property_assigner, instance_assigner, timing
    else:
        return writer.validate_and_write_ocel(oclog, config.output_dir, aug_log.name, config.resource_dir, config)




def check_multiple_roles(aug_log, role):
    return len([r for r in aug_log.attribute_to_concept_type.values() if r in type_mapping and type_mapping[r] == role]) > 0


def main():
    list_of_files = {}
    for (dir_path, dir_names, filenames) in os.walk(DEFAULT_CONFIG.input_dir):
        for filename in filenames:
            if filename.endswith('.xes') or (filename.endswith('.csv') and '_info' not in filename):
                list_of_files[filename] = os.sep.join([dir_path])
    print(list_of_files)
    for key, value in list_of_files.items():
        augment_and_transform_log(value, key, DEFAULT_CONFIG)


if __name__ == '__main__':
    main_tic = time.perf_counter()
    main()
    main_toc = time.perf_counter()
    print(f"Program finished all operations in {main_toc - main_tic:0.4f} seconds")
    sys.exit()
