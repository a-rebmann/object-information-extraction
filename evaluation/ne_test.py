import preprocessing.nlp_utils as nlputil
from preprocessing.preprocessor import preprocess_label

TEST_STRINGS = ["Echo", "Fire Stick 4K", "Echo Show 8", "iPad Pro", "Kindle", "Kindle Paperwhite", "Echo Dot",
                "MacBook Air", "MacBook Pro", "iPhone 11", "iPhone 11 Pro",
                "Marco Pegoraro", "Gyunam Park", "Majid Rafiei", "Junxiong Gao", "Seran Uysal", "Wil van der Aalst",
                "Han van der Aa", "Alexander Kraus", "Adrian Rebmann", "Jana-Rebecca Rehse", "Fareed Zandkarimi"]



if __name__ == '__main__':
    for s in TEST_STRINGS:
        print(nlputil.get_named_entities_if_present(s))