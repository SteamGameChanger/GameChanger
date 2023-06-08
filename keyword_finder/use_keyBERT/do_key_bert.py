import itertools
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class DoKeyBERT:
    def __init__(self, doc, n_gram_range):
        self.doc = doc
        self.n_gram_range = n_gram_range
        self.stop_words = 'english'

        self.count = None
        self.candidates = []

        self.model = None
        self.doc_embedding = None
        self.candidate_embeddings = None

        """
        top_n = [5, 10]
        nr_candidates = [10, 20]
        diversity = [0.2, 0.5, 0.7]
        """

    def vectorize(self):
        self.count = CountVectorizer(ngram_range=self.n_gram_range, stop_words=self.stop_words).fit([self.doc])
        self.candidates = self.count.get_feature_names_out()

    def embed(self):
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.doc_embedding = self.model.encode([self.doc])
        self.candidate_embeddings = self.model.encode(self.candidates)

    def max_sum_sim(self, top_n, nr_candidates):
        # 문서와 각 키워드들 간의 유사도
        distances = cosine_similarity(self.doc_embedding, self.candidate_embeddings)

        # 각 키워드들 간의 유사도
        distances_candidates = cosine_similarity(self.candidate_embeddings, 
                                                self.candidate_embeddings)

        # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
        words_idx = list(distances.argsort()[0][-nr_candidates:])
        words_vals = [self.candidates[index] for index in words_idx]
        distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

        # 각 키워드들 중에서 가장 덜 유사한 키워드들간의 조합을 계산
        min_sim = np.inf
        candidate = None
        for combination in itertools.combinations(range(len(words_idx)), top_n):
            sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
            if sim < min_sim:
                candidate = combination
                min_sim = sim

        return [words_vals[idx] for idx in candidate]

    def mmr(self, top_n, diversity):
        words = self.candidates
        
        # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
        word_doc_similarity = cosine_similarity(self.candidate_embeddings, self.doc_embedding)

        # 각 키워드들 간의 유사도
        word_similarity = cosine_similarity(self.candidate_embeddings)

        # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
        # 만약, 2번 문서가 가장 유사도가 높았다면
        # keywords_idx = [2]
        keywords_idx = [np.argmax(word_doc_similarity)]

        # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
        # 만약, 2번 문서가 가장 유사도가 높았다면
        # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

        # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
        # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
        for _ in range(top_n - 1):
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

            # MMR을 계산
            mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
            mmr_idx = candidates_idx[np.argmax(mmr)]

            # keywords & candidates를 업데이트
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        return [words[idx] for idx in keywords_idx]

    def run(self, top_n, nr_candidates, diversity):
        # Iterate over all parameter combinations
        for tn in top_n:
            for nc in nr_candidates:
                for d in diversity:

                    self.vectorize()
                    self.embed()

                    # Call max_sum_sim and mmr functions
                    print("Max Sum Similarity Keywords for top_n={}, nr_candidates={}".format(tn, nc))
                    print(self.max_sum_sim(tn, nc))
                    print("MMR Keywords for top_n={}, diversity={}".format(tn, d))
                    print(self.mmr(tn, d))
