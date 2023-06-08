from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.models.ctm import CombinedTM

class TopicModel:
    def __init__(self, num_topics, num_epochs, preprocessed_documents, unpreprocessed_corpus, contextual_model="paraphrase-distilroberta-base-v1"):
        self.tp = TopicModelDataPreparation(contextual_model)
        self.training_dataset = self.tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)
        self.ctm = CombinedTM(bow_size=len(self.tp.vocab), contextual_size=768, n_components=num_topics, num_epochs=num_epochs)

    def train(self):
        self.ctm.fit(self.training_dataset)

    def get_topic_predictions(self, n_samples=5):
        return self.ctm.get_thetas(self.training_dataset, n_samples=n_samples)

    def get_topic_lists(self, n_samples):
        return self.ctm.get_topic_lists(n_samples)

    def save_model(self, path):
        self.ctm.save(path)

    def load_model(self, path, epoch):
        self.ctm.load(path, epoch)
