from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing

class DataPreparation:
    def __init__(self, text_file, stopwords_language='english'):
        self.documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()]
        self.sp = WhiteSpacePreprocessing(self.documents, stopwords_language=stopwords_language)

    def preprocess(self):
        return self.sp.preprocess()
