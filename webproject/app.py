from flask import Flask, render_template, request

# keybert ------------------------------------------------
import itertools
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# XLNet ---------------------------------------------------
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import operator 

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# ---------------------------------------------------------

app = Flask(__name__)

# keybert--------------------------------
n_gram_range = (3,3)

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
        self.vectorize()
        self.embed()

        # Call max_sum_sim and mmr functions
        print("Max Sum Similarity Keywords for top_n={}, nr_candidates={}".format(top_n, nr_candidates))
        print(self.max_sum_sim(top_n, nr_candidates))
        print("MMR Keywords for top_n={}, diversity={}".format(top_n, diversity))
        print(self.mmr(top_n, diversity))

        return self.max_sum_sim(top_n, nr_candidates) + self.mmr(top_n, diversity)

# XLNet----------------------------------------
# kaggle data에서 수집한 장르 종류 (쥬피터에서 돌려서 긁어옴)

genres = ['Casual','Indie','Sports','Action','Adventure','Strategy','RPG','Simulation','Early Access','Racing','Massively Multiplayer','Utilities','Education','Sexual Content','Nudity','Violent','Gore','Web Publishing','Animation & Modeling','Design & Illustration','Free to Play','Software Training','Game Development','Photo Editing','Audio Production','Video Production','Accounting','Movie','Documentary','Episodic','Short','Tutorial','360 Video',
'Survival']

# 소문자 처
genres_lower = [g.lower() for g in genres]

# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
# If not...
else:
    device = torch.device("cpu")

def remove_stopwords(sentence):
    # 불용어 제거 
    stopwords = nltk.corpus.stopwords.words('english')
    stop_words = set(stopwords)
    words = word_tokenize(sentence)

    filtered_words = [word for word in words if word.casefold() not in stop_words]
    # 다시 문장으로 만들기 
    filtered_sentence = ' '.join(filtered_words)
    return filtered_sentence

def preprocess_text(text):
    # 특수 문자 제거
    text = re.sub(r'[^\w\s]', ' ', text)

    return text

def extract_keywords(text, top_k=30):

    text = remove_stopwords(text)
    text = preprocess_text(text)

    print("After Proprocessing: \n",text, '\n')

    # XLNet tokenizer 및 모델 로드
    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
    model = AutoModel.from_pretrained("xlnet-base-cased")

    model.to(device)
    torch.cuda.empty_cache()
    '''
    # 입력 텍스트 토큰화
    tokens = tokenizer.tokenize(text)

    input_ids = tokenizer.encode(text, return_tensors='pt') # XLNet에 맞게 속성 수정 
    input_ids = input_ids.to(device)
    '''
    tokens = tokenizer.tokenize(text)
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    # XLNet 모델 실행
    with torch.no_grad():
        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state 
    '''
    # XLNet 모델 실행
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        last_hidden_state = outputs.last_hidden_state
    '''
    # 키워드 추출
    # last_hidden_state[0].mean(dim=1) : 각 문장의 임베딩 벡터들을 평균화한 하나의 벡터 (해당 문장의 전체 문맥을 나타내는 벡터)
    #  ~.indices:  가장 큰 top_k개의 값들의 인덱스를 나타내는 텐서 (가장 중요한 토큰들의 위치를 확인할 수 있음)
    keyword_ids = torch.topk(last_hidden_state[0].mean(dim=1), top_k).indices.squeeze()
    keywords = [tokens[idx] for idx in set(keyword_ids)]

    return keywords

### 페이지 랜더링 리스트 
#  텍스트 입력 페이지(index.html) 
# 유사 게임 찾기 & 주요 키워드 찾기 결과 출력 페이지(result.html)
# 추후에 GET, POST 분기 작성
@app.route('/', methods=['GET', 'POST']) 
def index():
    return render_template("index.html")

@app.route('/result', methods=['GET','POST'])
def find():

    doc = request.form['text']
    # keybert--------------------------=====
    keyBERT = DoKeyBERT(doc, n_gram_range)
    top_n = 10
    nr_candidates = 20
    diversity = 0.5
    result = keyBERT.run(top_n, nr_candidates, diversity)

    # XLNet --------------------------------

    # 키워드
    final_key = []


    try: text_split, i = preprocess_text(doc).split(' '), 0
    except: text_split = doc

    while i < len(text_split):
        try:
            n = int(text_split[i][0])
            if len(text_split[i]) == 4: # 년도일 가능성 높음
                final_key.append(text_split[i])
                i += 1
            elif len(text_split[i+1]) > 2:
                final_key.append(text_split[i].lower() +' '+ text_split[i+1].lower())
            i += 2
        except:
            i += 1



    # 추출하기 & 단어만 골라내기
    top_k = 30
    while(True): # 만약 top_k개 뽑아낼 수 없는 경우 top_k를 줄여가며 시도
        if doc == 'nan': break
        try: 
            keywords_extracted = set(extract_keywords(doc, top_k))
            break
        except:
            top_k -= 5


    # XLNet 토크나이'▁' 때문에 명사 처럼 여겨지는 경우가 있어 처리
    keywords_extracted_copy = keywords_extracted.copy()
    for t in keywords_extracted_copy :
        if t == '▁':
            keywords_extracted.remove(t)
        elif t[0] == '▁':
            keywords_extracted.remove(t)
            t = t[1:]
            keywords_extracted.add(t)

    # 품사 태깅
    tokens_pos = nltk.pos_tag(keywords_extracted)


    # 명사, 형용사만 골라내기
    only_nORj = [] 
    for word, pos in tokens_pos:
        if 'NN' in pos  or 'JJ' in pos:
            w_lower = word.lower() # 소문자로 통일
            for gl in genres_lower: # 장르 관련 키워드면 무조건 포함
                if gl in w_lower:
                    final_key.append(w_lower)
                else: only_nORj.append(w_lower)

    only_nORj = set(only_nORj)
    # 여러 조건들에 맞춰 해당하는 것만 뽑아내기
    for w in only_nORj:
        if len(w) <= 3: continue # 3글자 이하 제거
        if w[-1] == 's' : # 끝에 s가 붙는 경우 s제거
            w = w[:-1]
        if ('game' in w) or ('thing' in w) or (w =='play') or w == 'fea': continue
        append_bool = True
        for k in final_key:
            if w in k:
                append_bool = False
                break
        if append_bool: final_key.append(w)

    print('\n\nKEY WORDS : ',set(final_key))

    #result += set(final_key)

    XL_result = ''
    for k in set(final_key):
        XL_result += k + '//'
    Bert_result = ''
    for k in result:
        Bert_result += k + '//'

    # --------------------------------------
    return render_template("result.html", text=doc, XL_result=XL_result,Bert_result = Bert_result)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
