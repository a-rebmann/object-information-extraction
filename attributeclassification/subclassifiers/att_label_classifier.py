from sklearn import feature_extraction, feature_selection
from pandas import DataFrame

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from const import consider_for_label_classification, class_labels
from data.labeled_data import LabeledData
from model.augmented_log import AugmentedLog
from preprocessing.preprocessor import clean_attribute_name
from sklearn.linear_model import SGDClassifier
from nltk import word_tokenize
import numpy as np
from readwrite.loader import deserialize_model
from readwrite.writer import serialize_model


def tfidf_glove(idf_dict, vals, glove):
    vectors = []
    for idx, val in enumerate(vals):
        glove_vectors = [glove[tok] if tok in glove.key_to_index.keys() else np.zeros(50) for tok in word_tokenize(val)]
        weights = [idf_dict.get(word, 1) if word in glove.key_to_index.keys() else 0.0000000001 for word in word_tokenize(val)]
        try:
            vectors.append(np.average(glove_vectors, axis=0, weights=weights))
        except ZeroDivisionError:
            print(val, "caused Zero Division error")
            vectors.append(np.zeros(50))
    return np.array(vectors)


class AttributeLabelClassifier:

    def __init__(self, config, data: LabeledData, aug_log: AugmentedLog):
        docs, concepts = data.get_attribute_data()
        print(len(docs), len(concepts))
        docs = [doc for i, doc in enumerate(docs) if concepts[i] in class_labels]
        concepts = [conc for conc in concepts if conc in class_labels]
        print(len(docs), len(concepts))
        self.config = config
        self.d = DataFrame({"text": docs, "y": concepts})
        self.cols = aug_log.get_attributes_by_att_types(consider_for_label_classification)

    def with_tf_idf_and_embedding(self, embeddings, eval_mode=False):
        res = {}
        md = deserialize_model(self.config.resource_dir, "att_class")

        if md is False or eval_mode is True:
            print("Build new attribute classifier")
            tfidf = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))
            tfidf.fit_transform(self.d["text"].values)
            # test = self.test["text"].values
            # Now lets create a dict so that for every word in the corpus we have a corresponding IDF value
            idf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
            x_train = tfidf_glove(idf_dict, self.d["text"].values, embeddings)
            #x_test = tfidf_glove(idf_dict, test, glove)
            enc = LabelEncoder()
            X = class_labels
            enc.fit(X)

            y_train = enc.transform(self.d["y"].values)
            #y_test = enc.transform(self.test["y"].values)
            clzz = build_log_reg(x_train, y_train)
            model_dict = {"idf_dict": idf_dict, "X": X, "enc": enc, "clzz": clzz}
            serialize_model(self.config.resource_dir, model_dict, "att_class")
        else:
            idf_dict = md["idf_dict"]
            enc = md["enc"]
            clzz = md["clzz"]
            X = md["X"]

        for plain in self.cols:
            x_t = tfidf_glove(idf_dict, [clean_attribute_name(plain)], embeddings)
            probas = clzz.predict_proba(x_t)[0]
            pred = enc.inverse_transform(clzz.predict(x_t))[0]
            #print(plain, pred, probas[X.index(pred)])
            res[plain] = pred, probas[X.index(pred)]

        #run_log_reg(x_train, x_test, y_train, y_test, enc)
        return res


    def with_tf_idf(self):
        vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        corpus = self.d["text"]
        vectorizer.fit(corpus)
        X_train = vectorizer.transform(corpus)
        y = self.d["y"]
        X_names = vectorizer.get_feature_names()
        p_value_limit = 0.85
        dtf_features = pd.DataFrame()
        for cat in np.unique(y):
            chi2, p = feature_selection.chi2(X_train, y == cat)
            dtf_features = dtf_features.append(pd.DataFrame(
                {"feature": X_names, "score": 1 - p, "y": cat}))
            dtf_features = dtf_features.sort_values(["y", "score"],
                                                    ascending=[True, False])
            dtf_features = dtf_features[dtf_features["score"] > p_value_limit]
        new_log_kb = {}
        for cat in np.unique(y):
            new_log_kb[cat] = dtf_features[dtf_features["y"] == cat]["feature"].values[:10]
            print("# {}:".format(cat))
            print("  . selected features:",
                  len(dtf_features[dtf_features["y"] == cat].values[:10]))
            print("  . top features:", ",".join(
                dtf_features[dtf_features["y"] == cat]["feature"].values[:10]))
            print(" ")
        return new_log_kb


def build_log_reg(train_features, y_train, alpha=1e-4):
    log_reg = SGDClassifier(loss='log', alpha=alpha, n_jobs=-1, penalty='l2')
    log_reg.fit(train_features, y_train)
    return log_reg