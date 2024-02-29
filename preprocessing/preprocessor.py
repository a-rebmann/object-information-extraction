import re
import time
from const import AttributeType, XES_NAME
from preprocessing.nlp_utils import get_named_entities_if_present, check_text, check_rich_text, \
    TERMS_FOR_MISSING
from multiprocessing import Pool
import numpy as np
import pandas as pd

CAMEL_PATTERN_1 = re.compile('(.)([A-Z][a-z]+)')
CAMEL_PATTERN_2 = re.compile('([a-z0-9])([A-Z])')

UUID_PATTERN = re.compile('\b[0-9a-f]{8}\b-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-\b[0-9a-f]{12}\b')

NON_ALPHANUM_PATTERN = re.compile('[^a-zA-Z]')

DATE_PATTERNS = [re.compile('([0-9]{4})[\\.\\/\\-]([0-9]{1,2})[\\.\\/\\-]([0-9]{1,2})'),
                 re.compile(
                     '^(?=\\d)(?:(?!(?:(?:0?[5-9]|1[0-4])(?:\\.|-|\\/)10(?:\\.|-|\\/)(?:1582))|(?:(?:0?[3-9]|1[0-3])(?:\\.|-|\\/)0?9(?:\\.|-|\\/)(?:1752)))(31(?!(?:\\.|-|\\/)(?:0?[2469]|11))|30(?!(?:\\.|-|\\/)0?2)|(?:29(?:(?!(?:\\.|-|\\/)0?2(?:\\.|-|\\/))|(?=\\D0?2\\D(?:(?!000[04]|(?:(?:1[^0-6]|[2468][^048]|[3579][^26])00))(?:(?:(?:\\d\\d)(?:[02468][048]|[13579][26])(?!\\x20BC))|(?:00(?:42|3[0369]|2[147]|1[258]|09)\\x20BC))))))|2[0-8]|1\\d|0?[1-9])([-.\\/])(1[012]|(?:0?[1-9]))\\2((?=(?:00(?:4[0-5]|[0-3]?\\d)\\x20BC)|(?:\\d{4}(?:$|(?=\\x20\\d)\\x20)))\\d{4}(?:\\x20BC)?)(?:$|(?=\\x20\\d)\\x20))?((?:(?:0?[1-9]|1[012])(?::[0-5]\\d){0,2}(?:\\x20[aApP][mM]))|(?:[01]\\d|2[0-3])(?::[0-5]\\d){1,2})?$'),
                 re.compile(
                     '^(?:(?:(?:(?:(?:1[6-9]|[2-9]\\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00)))(\\/|-|\\.)(?:0?2\\1(?:29)))|(?:(?:(?:1[6-9]|[2-9]\\d)?\\d{2})(\\/|-|\\.)(?:(?:(?:0?[13578]|1[02])\\2(?:31))|(?:(?:0?[1,3-9]|1[0-2])\\2(29|30))|(?:(?:0?[1-9])|(?:1[0-2]))\\2(?:0?[1-9]|1\\d|2[0-8]))))$'),
                 re.compile(
                     '^(((0?[1-9]|[12]\\d|3[01])[\\.\\-\\/](0?[13578]|1[02])[\\.\\-\\/]((1[6-9]|[2-9]\\d)?\\d{2}))|((0?[1-9]|[12]\\d|30)[\\.\\-\\/](0?[13456789]|1[012])[\\.\\-\\/]((1[6-9]|[2-9]\\d)?\\d{2}))|((0?[1-9]|1\\d|2[0-8])[\\.\\-\\/]0?2[\\.\\-\\/]((1[6-9]|[2-9]\\d)?\\d{2}))|(29[\\.\\-\\/]0?2[\\.\\-\\/]((1[6-9]|[2-9]\\d)?(0[48]|[2468][048]|[13579][26])|((16|[2468][048]|[3579][26])00)|00)))$'),
                 re.compile('^([0]?[1-9]|[1][0-2])[./-]([0]?[1-9]|[1|2][0-9]|[3][0|1])[./-]([0-9]{4}|[0-9]{2})$')]


def pre_process(config, ld, consider_case_attributes=True, sample_for_annotation=500, max_unique_vals=10000,
                allowed_propn_ratio=0.2):
    """
    Takes a LogDescriptor and removes all camel case & underscores.
    Optionally converts all attributes to lower case and marks natural language
    columns with AttributeType RICH_TEXT in descriptor
    """
    tic = time.perf_counter()
    df = ld.df
    if not df.size <= 1:
        duplicates = get_duplicate_columns(df)

        ld.duplicates = duplicates
        if len(duplicates) > 0:
            print("Found " + str(len(duplicates)) + " duplicate columns" + ". Removing them and keeping names in mind.")
        to_drop = []
        for val in duplicates.values():
            for c in val:
                to_drop.append(c)
        for c in df.columns:
            if "Unnamed:" in c:
                to_drop.append(c)
        df.drop(to_drop, axis=1, inplace=True)
        toc = time.perf_counter()
        print(f"Removed duplicates after {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    if sample_for_annotation > len(df.index):
        sample_for_annotation = len(df.index)
    dt = df.dtypes
    print(dt)

    for att in df.columns:
        if (not consider_case_attributes) and (att in ld.case_attributes or att == ld.case_id):
            ld.set_attribute_type(att, AttributeType.CASE_ATT)
            continue
        ld.num_uniques[att] = df[att].nunique()
        if dt[att] == 'int64':
            ld.set_attribute_type(att, AttributeType.INT)
        elif dt[att] == 'float64' or str(dt[att]) == 'timedelta[ns]':
            if np.array_equal(df[att].dropna(), df[att].dropna().astype(int)):
                ld.set_attribute_type(att, AttributeType.INT)
            else:
                ld.set_attribute_type(att, AttributeType.NUMERIC)
        elif dt[att] == 'bool':
            ld.set_attribute_type(att, AttributeType.FLAG)
        elif dt[att] == 'datetime64' or 'datetime64' in str(dt[att]) or df.sample(n=sample_for_annotation).apply(
                lambda x: check_for_date(str(x[att])), axis=1).all():
            ld.set_attribute_type(att, AttributeType.TIMESTAMP)
        elif dt[att] == 'object':
            if df.sample(n=sample_for_annotation).apply(
                    lambda x: all([s.isdigit() for s in str(x)]), axis=1).all():
                ld.set_attribute_type(att, AttributeType.INT)
            elif df.sample(n=sample_for_annotation).apply(
                    lambda x: len(preprocess_label(str(x[att]))) < 2, axis=1).all():

                ld.set_attribute_type(att, AttributeType.STRING)
    toc = time.perf_counter()
    print(f"Annotated primitive type columns in {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    filtered_df = ld.get_df_representation_filtered(AttributeType.UNKNOWN, True).fillna('')
    toc = time.perf_counter()
    print(f"Obtained a filtered copy of the log in {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    for col in filtered_df.columns:
        unique_values = filtered_df[col].unique().tolist()
        unique = len(unique_values)
        if unique < 2:
            ld.set_attribute_type(col, AttributeType.STRING)
            ld.att_to_unique[col] = unique_values
            continue
        # TODO large amount of data values
        if unique > ld.get_num_cases() > max_unique_vals:
            print("too many unique values in", col)
            ld.set_attribute_type(col, AttributeType.STRING)
            ld.att_to_unique[col] = unique_values
            continue

        # We need to handle named entity recognition before pre processing strings!
        count_gpe = 0.0
        for lab in unique_values:
            if not isinstance(lab, str):
                continue
            nes = get_named_entities_if_present(lab)
            if len(nes) > 0:
                for entry in nes:
                    if entry[3] == 'PERSON' or entry[3] == 'ORG' or entry[3] == 'GPE' or entry[3] == 'LAW' or entry[
                        3] == 'PRODUCT' or entry[3] == 'WORK_OF_ART':
                        count_gpe += 1.0

        if count_gpe / len(unique_values) > config.instance_thresh and col != XES_NAME:
            print("There are named entities in ", col)
            ld.attribute_to_ne[col] = "NE"
            ld.set_attribute_type(col, AttributeType.STRING)
            continue
        filtered_df[col] = parallelize_dataframe(filtered_df[col], filter_col)
        unique_values = [str(i) for i in filtered_df[col].unique().tolist() if
                         i not in TERMS_FOR_MISSING and not pd.isna(i)]
        if filtered_df.sample(n=sample_for_annotation).apply(lambda x: len(preprocess_label(str(x[col]))) < 2,
                                                             axis=1).all():
            ld.set_attribute_type(col, AttributeType.STRING)
            ld.att_to_unique[col] = unique_values
            print("only one value after preprocessing text", col)
            continue
        b = filtered_df.sample(n=sample_for_annotation).apply(lambda x: check_text(str(x[col])), axis=1).mean() > 0.1
        if b:
            if len(unique_values) <= 2 and ("false" in unique_values or "true" in unique_values):
                ld.set_attribute_type(col, AttributeType.FLAG)
                ld.att_to_unique[col] = unique_values
                continue
            if len(unique_values) < 2:
                print('skipping ' + col + ' due to constant value')
                ld.set_attribute_type(col, AttributeType.STRING)
                ld.att_to_unique[col] = unique_values
                continue
            recognize_ne(unique_values, ld)
            ld.att_to_unique[col] = unique_values
            if filtered_df.sample(n=sample_for_annotation).apply(lambda x: check_rich_text(str(x[col])),
                                                                 axis=1).mean() > allowed_propn_ratio:
                ld.set_attribute_type(col, AttributeType.RICH_TEXT)
            else:
                print("Rich text check failed")
                ld.set_attribute_type(col, AttributeType.STRING)
        else:
            print('after checking POS, no solid amount of proper words was found in ' + col)
            ld.set_attribute_type(col, AttributeType.STRING)
            ld.att_to_unique[col] = unique_values
    toc = time.perf_counter()
    print(f"String operations on the log took {toc - tic:0.4f} seconds")
    ld.set_cleaned_df(filtered_df)
    return ld


def _camel_to_white(label):
    label = CAMEL_PATTERN_1.sub(r'\1 \2', label)
    return CAMEL_PATTERN_2.sub(r'\1 \2', label)


def check_for_and_remove_dates(li: list):
    res = []
    for item in li:
        for pattern in DATE_PATTERNS:
            if pattern.search(item):
                print('found date')
                continue
            else:
                res.append(item)
    return res


def recognize_ne(list_of_sents, ld):
    for sent in list_of_sents:
        sent = str(sent)
        ne_tagging = get_named_entities_if_present(sent)
        if len(ne_tagging) > 0:
            ld.add_ne_for_label(sent, ne_tagging)


def check_for_date(att):
    for pattern in DATE_PATTERNS:
        if pattern.search(att):
            return True
    return False


def check_for_uuid(att):
    if UUID_PATTERN.search(att):
        return True
    return False


def preprocess_label(label):
    label = str(label)
    label = _camel_to_white(label).lower()
    label = NON_ALPHANUM_PATTERN.sub(' ', label)
    parts = label.split()
    res = []
    for part in parts:
        clean = ''.join([i for i in part if not i.isdigit()])
        res.append(clean)
    return ' '.join(res)


def filter_col(column):
    return column.astype(str).str.replace(
        CAMEL_PATTERN_1, r'\1 \2', regex=True).str.replace(
        CAMEL_PATTERN_2, r'\1 \2', regex=True).str.lower().str.replace(
        NON_ALPHANUM_PATTERN, ' ', regex=True).str.replace(
        ' +', ' ', regex=True).str.strip().str.replace(
        '\d+', '', regex=True).str.strip()


def clean_attribute_name(att_name):
    return preprocess_label(att_name).replace('case', '').lower().strip()


def clean_attribute_name_for_object(att_name):
    return clean_attribute_name(
        att_name).replace(" type", "").replace(" category", "").strip()


def parallelize_dataframe(df, func, n_cores=8):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def get_duplicate_columns(df):
    # Create an empty set
    duplicate_columns = dict()

    # Iterate through all the columns
    # of dataframe
    for x in range(df.shape[1]):

        # Take column at xth index.
        col = df.iloc[:, x]

        # Iterate through all the columns in
        # DataFrame from (x + 1)th index to
        # last index
        for y in range(x + 1, df.shape[1]):

            # Take column at yth index.
            otherCol = df.iloc[:, y]

            # Check if two columns at x & y
            # index are equal or not,
            # if equal then adding
            # to the set
            if col.equals(otherCol):
                if df.columns.values[y] == "case:concept:name":
                    if df.columns.values[y] not in duplicate_columns.keys():
                        duplicate_columns[df.columns.values[y]] = list()
                    duplicate_columns[df.columns.values[y]].append(df.columns.values[x])
                elif df.columns.values[x] == "case:concept:name":
                    if df.columns.values[x] not in duplicate_columns.keys():
                        duplicate_columns[df.columns.values[x]] = list()
                    duplicate_columns[df.columns.values[x]].append(df.columns.values[y])
                elif df.columns.values[y] == "concept:name":
                    if df.columns.values[y] not in duplicate_columns.keys():
                        duplicate_columns[df.columns.values[y]] = list()
                    duplicate_columns[df.columns.values[y]].append(df.columns.values[x])
                elif df.columns.values[x] == "concept:name":
                    if df.columns.values[x] not in duplicate_columns.keys():
                        duplicate_columns[df.columns.values[x]] = list()
                    duplicate_columns[df.columns.values[x]].append(df.columns.values[y])
                else:
                    if df.columns.values[x] not in duplicate_columns.keys():
                        duplicate_columns[df.columns.values[x]] = list()
                    duplicate_columns[df.columns.values[x]].append(df.columns.values[y])
    return duplicate_columns
