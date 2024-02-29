def data_to_object(dataframe, cols=False, only_label=[]):
    objects = []
    for index, row in dataframe.iterrows():
        if not cols:
            objects.append(DataObject(row))
        else:
            for col in dataframe.columns:
                if (len(only_label) == 0) and ('_tags' in col):
                    objects.append(DataObject(row, col))
                elif len(only_label) > 0:
                    for consider in only_label:
                        if consider in col:
                            objects.append(DataObject(row, col))
    return objects


def parse_tags(item, col):
    w, t = split_tags(item[col])
    return w, t


def split_tags(tags):
    w = []
    t = []
    tag_sets = tags.split(',')
    tag_sets = tag_sets[0:-1]
    for tag_set in tag_sets:
        word = tag_set.split('<>')[0].strip(' ')
        tag = tag_set.split('<>')[1].strip(' ')
        w.append(word)
        t.append(tag)
    return w, t


def to_list_format(train_objects):
    label_to_list_tuple = {}
    for item in train_objects:
        label_to_list_tuple[item.label] = (item.split.copy(), item.tags.copy())
    return label_to_list_tuple


def to_tuple_format(train_objects):
    train_sentences_semantic = {}
    for item in train_objects:
        curr = []
        for i in range(len(item.split)):
            curr.append((item.split[i], item.tags[i]))
        train_sentences_semantic[' '.join(item.split)] = curr
    return train_sentences_semantic


class DataObject:
    def __init__(self, item, col='Tags'):
        self.split, self.tags = parse_tags(item, col)
        lab = ''
        for part in self.split:
            lab += part + ' '
        self.label = lab.strip()


def prepare_data(train_objects):
    train_sentences = []
    for item in train_objects:
        curr = []
        for i in range(len(item.split)):
            curr.append((item.split[i], item.tags[i]))
        train_sentences.append(curr)


def prepare_tag_set(train_sentences):
    tags = set([item for sublist in train_sentences for _, item in sublist])
    tag2idx = {}
    idx2tag = {}
    for i, tag in enumerate(sorted(tags)):
        tag2idx[tag] = i + 1
        idx2tag[i + 1] = tag

    # Special character for the tags
    tag2idx['[PAD]'] = 0
    idx2tag[0] = '[PAD]'


def _tag_sequence(sentences):
    return [[t for w, t in sentence] for sentence in sentences]


def _text_sequence(sentences):
    return [[w for w, t in sentence] for sentence in sentences]
