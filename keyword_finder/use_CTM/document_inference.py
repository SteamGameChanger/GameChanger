from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing

class DocumentInference:
    def __init__(self, topic_model, documents):
        self.topic_model = topic_model
        self.documents = documents

    def preprocess_documents(self):
        sp = WhiteSpacePreprocessing(self.documents, stopwords_language='english')
        return sp.preprocess()

    def infer_topics(self):
        preprocessed_documents, unpreprocessed_corpus, vocab = self.preprocess_documents()
        test_dataset = self.topic_model.tp.transform(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)
        topic_predictions = self.topic_model.get_topic_predictions(test_dataset)
        return topic_predictions
