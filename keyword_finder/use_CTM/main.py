from data_preparation import DataPreparation
from topic_model import TopicModel
import os

os.chdir("C:\\Users\\JongbeenSong\\Desktop\\23-1\\빅데이터 처리\\BDP-main\\keyword_finder\\use_CTM")

# Preparation
dp = DataPreparation(text_file="dbpedia_sample_abstract_20k_unprep.txt")
preprocessed_documents, unpreprocessed_corpus, vocab = dp.preprocess()

# Training
if __name__ == '__main__':
    print('running')
    tm = TopicModel(num_topics=50, num_epochs=20, preprocessed_documents=preprocessed_documents, unpreprocessed_corpus=unpreprocessed_corpus)
    tm.train()

    # Evaluation
    topic_predictions = tm.get_topic_predictions()
    topic_lists = tm.get_topic_lists(5)

    # Save
    tm.save_model("./")
    del tm
