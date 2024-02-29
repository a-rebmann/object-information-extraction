#import wikipediaapi
import spacy

nlp = spacy.load("en_core_web_lg")

STOP_WORDS = ["i",
              "me",
              "my",
              "myself",
              "we",
              "our",
              "ours",
              "ourselves",
              "you",
              "your",
              "yours",
              "yourself",
              "yourselves",
              "he",
              "him",
              "his",
              "himself",
              "she",
              "her",
              "hers",
              "herself",
              "it",
              "its",
              "itself",
              "they",
              "them",
              "their",
              "theirs",
              "themselves",
              "what",
              "which",
              "who",
              "whom",
              "this",
              "that",
              "these",
              "those",
              "am",
              "is",
              "are",
              "was",
              "were",
              "be",
              "been",
              "being",
              "have",
              "has",
              "had",
              "having",
              "do",
              "does",
              "did",
              "doing",
              "a",
              "an",
              "the",
              "and",
              "but",
              "if",
              "or",
              "because",
              "as",
              "until",
              "while",
              "of",
              "at",
              "by",
              "for",
              "with",
              "about",
              "against",
              "between",
              "into",
              "through",
              "during",
              "before",
              "after",
              "above",
              "below",
              "to",
              "from",
              "up",
              "down",
              "in",
              "out",
              "on",
              "off",
              "over",
              "under",
              "again",
              "further",
              "then",
              "once",
              "here",
              "there",
              "when",
              "where",
              "why",
              "how",
              "all",
              "any",
              "both",
              "each",
              "few",
              "more",
              "most",
              "other",
              "some",
              "such",
              "no",
              "nor",
              "not",
              "only",
              "own",
              "same",
              "so",
              "than",
              "too",
              "very",
              "s",
              "t",
              "can",
              "will",
              "just",
              "don",
              "should",
              "now"]

TERMS_FOR_MISSING = ['undefined', 'missing', 'none', 'nan', 'empty', 'empties', 'unknown', 'other', 'others', 'na',
                     'nil', 'null', '', ' ']


def get_pos(plain_text):
    doc = nlp(plain_text)
    pos = [tok.pos_ for tok in doc]
    return pos


def check_text(plain_text):
    doc = nlp(plain_text)
    pos = [tok.pos_ for tok in doc]
    # print(plain_text, pos)
    if 'NOUN' in pos or 'VERB' in pos or 'PROPN' in pos or 'ADJ' in pos:
        return True
    else:
        return False


def check_rich_text(plain_text):
    doc = nlp(plain_text)
    pos = [tok.pos_ for tok in doc]
    if 'NOUN' in pos or 'VERB' in pos or 'ADV' in pos or 'ADJ' in pos:
        return True
    else:
        return False


def check_propn(tok):
    doc = nlp(tok)
    pos = [tok.pos_ for tok in doc]
    if all(p == 'PROPN' for p in pos):
        return True
    else:
        return False


def tokenize(plain_text):
    """
    retruns spaCy specific representation
    => token (token.i, token.text, token.is_alpha, token.like_num)
    :param plain_text:
    :return: tokenized text
    """
    return nlp.tokenizer(plain_text)


def lemmatize(plain_text):
    """
    retruns spaCy specific representation
    => token (token.i, token.text, token.is_alpha, token.like_num)
    :param plain_text:
    :return: tokenized text
    """
    doc = nlp(plain_text)
    ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
    return ents


def get_semantic_similarity(val1, val2):
    tokens1 = nlp(val1)
    tokens2 = nlp(val2)
    sim = 0.0
    i = 0
    max_sim = 0
    for tok1 in tokens1:
        for tok2 in tokens2:
            new_sim = tok1.similarity(tok2)
            if new_sim > max_sim:
                max_sim = new_sim
            sim += new_sim
            i += 1
    if i == 0:
        return 0
    return sim / i


def get_named_entities_if_present(label: str):
    try:
        doc = nlp(label)
        ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
    except ValueError:
        print(label)
        ents = []
    # if len(ents) > 0: print(ents)
    return ents


# def wiki_lookup(string):
#     wiki_wiki = wikipediaapi.Wikipedia('en')
#     page_py = wiki_wiki.page(string)
#     return page_py.exists()
