import csv
import json
import os
from const import ConceptType
import pickle
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import ocel


def write_trace_bo_lifecycle_summary(directory, the_dict):
    print('writing the life cycles to file')
    with open(os.path.join(directory, 'trace_bo_lifecycle_summary.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["case", "bo", "action_sequence", "actor_sequence", "receiver_sequence"])
        for case_id in the_dict.keys():
            for bo in the_dict[case_id].keys():
                writer.writerow([case_id, bo, the_dict[case_id][bo][ConceptType.ACTION_NAME.value],
                                 the_dict[case_id][bo][ConceptType.ACTOR_NAME.value],
                                 the_dict[case_id][bo][ConceptType.PASSIVE_NAME.value]])


def write_masked_results(directory, log, the_dict):
    print('writing sents to file')
    with open(os.path.join(directory, 'sentences.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["log", "value", "component", "sentence"])
        for val in the_dict.keys():
            for bo in the_dict[val].keys():
                writer.writerow([log, val, bo, the_dict[val][bo]])


def write_to_tagging_format(the_dict: dict, directory, name):
    print('writing the typeandactionextraction set to file')
    with open(os.path.join(directory, name+'.csv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Label", 'Tags'])
        for label in the_dict.keys():
            tagging = ''
            for index, val in enumerate(the_dict[label][0]):
                tagging = tagging + val + '<>' + the_dict[label][1][index] +', '
            writer.writerow([label, tagging])


def export_kb(directory, name, kb):
    outfile = open(os.path.join(directory, name), 'wb')
    pickle.dump(kb, outfile)
    outfile.close()


def create_file_for_df(df, direct, name='default'):
    if '.csv' in name:
        name = name.replace(".csv", "_augmented.csv")
    elif '.xes' in name:
        name = name.replace(".xes", "_augmented.csv")
    df.to_csv(direct + name, sep=';')


def write_event_log(log, direct, name):
    if '.csv' in name:
        name = name.replace(".csv", "_augmented.xes")
    elif '.xes' in name:
        name = name.replace(".xes", "_augmented.xes")
    xes_exporter.apply(log, direct+name)


def validate_and_write_ocel(oclog, direct, name, schema_path, config):
    if len(config.mask_attribute) != 0:
        name += str(config)
    ocel.export_log(oclog, direct + name + ".jsonocel")
    return name + ".jsonocel"
    # if ocel.validate(direct + name + ".jsonocel", schema_path + "schema.json"):
    #     return name + ".jsonocel"
    # else:
    #     os.remove(direct + name+ ".jsonocel")
    #     print(Warning("The OCEL-file was invalid and, thus, deleted."))
    #     return ""

def serialize_event_log(path, aug_log):
    with open(os.path.join(path, aug_log.name + '.pkl'), 'wb') as f:
        pickle.dump(aug_log, f)


def serialize_model(path, model, name):
    with open(os.path.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(model, f)


def serialize_data(path, data, name):
    with open(os.path.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(data, f)


def serialize_new_log(resource_dir, log):
    with open(os.path.join(resource_dir, log.name + '_t.pkl'), 'wb') as f:
        pickle.dump(log, f)


def serialize_eval_result(output_dir, current_config, filename, data):
    with open(os.path.join(output_dir, str(current_config)+filename + "_eval" + '.pkl'), 'wb') as f:
        pickle.dump(data, f)


def serialize_oc_dict(output_dir, current_config, filename, oclog):
    with open(os.path.join(output_dir, str(current_config)+filename + "_eval" + '.pkl'), 'wb') as f:
        pickle.dump(oclog, f)