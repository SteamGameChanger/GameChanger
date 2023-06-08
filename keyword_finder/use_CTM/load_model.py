from keyword_finder.use_CTM.topic_model import TopicModel

tm = TopicModel(num_topics=50, num_epochs=20, preprocessed_documents=preprocessed_documents, unpreprocessed_corpus=unpreprocessed_corpus)
tm.load_model("path_to_your_model", epoch=19)
