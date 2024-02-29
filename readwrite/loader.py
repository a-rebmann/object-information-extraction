import os
import json
import pickle
from datetime import timedelta

import ocel
import pandas as pd
import data.gathering.data_generator as fsg
from const import ConceptType, XES_CASE, XES_TIME, XES_NAME
from pm4py.objects.log.importer.xes import importer
from pm4py.objects.conversion.log import converter
from pm4py.objects.log.util import dataframe_utils

KNOWN_LOG_KEYS = {'Detail_Incident_Activity.csv': 'Incident ID'}

KNOWN_TIMESTAMP_KEYS = {'Detail_Incident_Activity.csv': 'DateStamp'}

KNOWN_EVENT_LABEL_KEYS = {'Detail_Incident_Activity.csv': 'IncidentActivity_Type'}


def handle_time_stamp_equality(df, time_stamp):
    for stamp, group in df.groupby(time_stamp):
        if len(group) > 1:
            i = 0
            for idx, member in group.iterrows():
                print(idx)
                df.loc[idx, time_stamp] = df.loc[idx, time_stamp] + timedelta(milliseconds=i)
                i += 1
            print("-----")


def handle_case_attributes(df):
    for col in df.columns:
        if all(len(group[col].dropna().unique()) == 1 for case, group in df.groupby(XES_CASE)):
            for case, group in df.groupby(XES_CASE):
                val = list(group[col].dropna().unique())[0]
                #print(val)
                for idx, member in group.iterrows():
                    df.loc[idx, col] = val
    return df

def load_log(filepath, filename, log_keys=KNOWN_LOG_KEYS, time_keys=KNOWN_TIMESTAMP_KEYS,
             event_label_keys=KNOWN_EVENT_LABEL_KEYS):
    df = {}
    if filename.endswith('.xes'):
        log = importer.apply(os.path.join(filepath, filename))
        df = converter.apply(log, variant=converter.Variants.TO_DATA_FRAME)
        try:
            # df = dataframe_utils.convert_timestamp_columns_in_df(df)
            pass
        except TypeError:
            pass
    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(os.path.join(filepath, filename), sep=';')
        except UnicodeDecodeError:
            df = pd.read_csv(os.path.join(filepath, filename), sep=';', encoding="ISO-8859-1")
        try:
            df = dataframe_utils.convert_timestamp_columns_in_df(df, timest_columns=[XES_TIME])
        except TypeError:
            print("conversion error")
            pass
    if filename.endswith('ocel'):
        ocel = load_ocel(filepath, filename)
        d = {att: [] for att in
             ocel["ocel:global-log"]["ocel:attribute-names"] + ocel["ocel:global-log"]["ocel:object-types"]}
        df = pd.DataFrame(data=d)
        print(df.columns)
        df[XES_TIME] = []
        df[XES_NAME] = []
        df[XES_CASE] = []
    case_id = log_keys[filename] if filename in log_keys else XES_CASE
    event_label = event_label_keys[filename] if filename in event_label_keys else XES_NAME
    time_stamp = time_keys[filename] if filename in time_keys else XES_TIME
    if filename in log_keys:
        case_id = log_keys[filename]
    if XES_TIME in df.columns:
        df[XES_TIME] = pd.to_datetime(df[XES_TIME], utc=True)
    elif filename in time_keys:
        df[XES_TIME] = pd.to_datetime(df[time_keys[filename]], utc=True)
        df[time_keys[filename]] = pd.to_datetime(df[time_keys[filename]], utc=True)
    if "item" in filename and "csv" in filename: #TODO
        df = handle_case_attributes(df)
    #handle_time_stamp_equality(df, time_stamp)
    return df, case_id, event_label, time_stamp


def load_ocel(filepath, filename):
    return ocel.import_log(filepath + filename)


def convert_df_to_log(df, filename, log_keys=KNOWN_LOG_KEYS):
    if filename in log_keys:
        return converter.apply(df, parameters={converter.to_event_log.Parameters.CASE_ID_KEY: log_keys[filename]},
                               variant=converter.Variants.TO_EVENT_LOG)
    return converter.apply(df, variant=converter.Variants.TO_EVENT_LOG)


def get_kb_true_labels(direct, file, log):
    ents = {ConceptType.ACTOR_NAME.value: set(), ConceptType.ACTION_NAME.value: set(),
            ConceptType.PASSIVE_NAME.value: set(),
            ConceptType.BO_NAME.value: set(),
            ConceptType.OTHER.value: set(), ConceptType.ACTION_STATUS.value: set(), ConceptType.BO_STATUS.value: set()}
    log_kb = {}
    with open(os.path.join(direct, file)) as json_file:
        kb = json.load(json_file)
        for log_name in kb.keys():
            if log_name == log:
                for entity_type in ents.keys():
                    for att in kb[log_name][entity_type]:
                        log_kb[att] = entity_type
    return log_kb


def get_annotated_attributes(direct, file):
    with open(os.path.join(direct, file)) as json_file:
        data = json.load(json_file)
        return data


def get_tagged_labels(direct, file):
    df = pd.read_csv(os.path.join(direct, file), sep=';', keep_default_na=False)
    entities_from_model = fsg.DataGenerator(df)
    return entities_from_model


def deserialize_event_log(path, name):
    try:
        with open(os.path.join(path, name + '.pkl'), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return False


def deserialize_model(path, name):
    try:
        with open(os.path.join(path, name + '.pkl'), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return False


def deserialize_data(path, name):
    try:
        with open(os.path.join(path, name + '.pkl'), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return False


def deserialize_new_log(path, name):
    try:
        with open(os.path.join(path, name + '_t.pkl'), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return False


def load_previous_result(path, current_config, filename):
    try:
        with open(os.path.join(path, str(current_config) + filename + "_eval" + '.pkl'), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return False
