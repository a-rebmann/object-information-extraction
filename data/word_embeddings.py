import gensim.downloader as api


class WordEmbeddings:

    def __init__(self, config):
        self.config = config
        self.embeddings = api.load(self.config.word_embeddings_file)
