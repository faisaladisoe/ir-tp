import random
import subprocess
import numpy as np
import lightgbm as lgb

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine

def data_preparation():
  proc = subprocess.Popen('wget -c https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz -P data'.split())
  proc.wait()
  proc = subprocess.Popen('tar -xvf data/nfcorpus.tar.gz'.split())
  proc.wait()

class DataInitialization():
  documents = {}
  queries = {}
  q_docs_rel = {}
  group_qid_count = []
  dataset = []

  def __init__(self, docs, queries, qrels):
    self.documents = self.init_documents(docs)
    self.queries = self.init_queries(queries)
    self.q_docs_rel = self.map_qrel_and_qid(qrels)
    self.group_qid_count, self.dataset = self.map_total_qid_and_dataset(self.q_docs_rel)

  def init_documents(self, docs):
    documents = {}
    with open(docs) as file:
      for line in file:
        doc_id, content = line.split("\t")
        documents[doc_id] = content.split()
    return documents

  def init_queries(self, queries):
    inner_queries = {}
    with open(queries) as file:
      for line in file:
        q_id, content = line.split("\t")
        inner_queries[q_id] = content.split()
    return inner_queries

  def map_qrel_and_qid(self, qrels):
    q_docs_rel = {} # grouping by q_id terlebih dahulu
    with open(qrels) as file:
      for line in file:
        q_id, _, doc_id, rel = line.split("\t")
        if (q_id in self.queries) and (doc_id in self.documents):
          if q_id not in q_docs_rel:
            q_docs_rel[q_id] = []
          q_docs_rel[q_id].append((doc_id, int(rel)))
    return q_docs_rel

  def map_total_qid_and_dataset(self, q_docs_rel):
    NUM_NEGATIVES = 1

    # group_qid_count untuk model LGBMRanker
    group_qid_count = []
    dataset = []
    for q_id in q_docs_rel:
      docs_rels = q_docs_rel[q_id]
      group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
      for doc_id, rel in docs_rels:
        dataset.append((self.queries[q_id], self.documents[doc_id], rel))
      # tambahkan satu negative (random sampling saja dari documents)
      dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))
    return group_qid_count, dataset
  
  def getter(self, component):
    if component == 'dataset':
      return self.dataset
    elif component == 'group_qid_count':
      return self.group_qid_count
    elif component == 'documents':
      return self.documents
    elif component == 'queries':
      return self.queries
    elif component == 'q_docs_rel':
      return self.q_docs_rel

class LSIAModel():
  # bentuk dictionary, bag-of-words corpus, dan kemudian Latent Semantic Indexing
  # dari kumpulan 3612 dokumen.
  NUM_LATENT_TOPICS = 200
  model = None
  queries = None
  documents = None
  data_init = None
  dictionary = None
  model_ranker = None

  def __init__(self):
    self.data_init = DataInitialization('./nfcorpus/train.docs', './nfcorpus/train.vid-desc.queries', './nfcorpus/train.3-2-1.qrel')
    self.documents = self.data_init.getter('documents')
    self.queries = self.data_init.getter('queries')

    self.dictionary = Dictionary()
    bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]
    self.model = LsiModel(bow_corpus, num_topics = self.NUM_LATENT_TOPICS) # 200 latent topics
  
  # test melihat representasi vector dari sebuah dokumen & query
  def vector_rep(self, text):
    rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
    return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS
  
  def features(self, query, doc):
    # kita ubah dataset menjadi terpisah X dan Y
    # dimana X adalah representasi gabungan query+document,
    # dan Y adalah label relevance untuk query dan document tersebut.
    # 
    # Bagaimana cara membuat representasi vector dari gabungan query+document?
    # cara simple = concat(vector(query), vector(document)) + informasi lain
    # informasi lain -> cosine distance & jaccard similarity antara query & doc
    v_q = self.vector_rep(query)
    v_d = self.vector_rep(doc)
    q = set(query)
    d = set(doc)
    cosine_dist = cosine(v_q, v_d)
    jaccard = len(q & d) / len(q | d)
    return v_q + v_d + [jaccard] + [cosine_dist]
  
  def map_to_X_and_Y(self):
    X = []
    Y = []
    for (query, doc, rel) in self.data_init.getter('dataset'):
      X.append(self.features(query, doc))
      Y.append(rel)

    # ubah X dan Y ke format numpy array
    X = np.array(X)
    Y = np.array(Y)

    return X, Y
  
  def train_the_model(self):
    X, Y = self.map_to_X_and_Y()
    ranker = lgb.LGBMRanker(
                    objective="lambdarank",
                    boosting_type = "gbdt",
                    n_estimators = 100,
                    importance_type = "gain",
                    metric = "ndcg",
                    num_leaves = 40,
                    learning_rate = 0.02,
                    max_depth = -1)

    # di contoh kali ini, kita tidak menggunakan validation set
    # jika ada yang ingin menggunakan validation set, silakan saja
    ranker.fit(X, Y,
              group = self.data_init.getter('group_qid_count'),
              verbose = 10)
    self.model_ranker = ranker
    return ranker
  
  def predict_model(self, query, docs):
    # bentuk ke format numpy array
    X_unseen = []
    for doc_id, doc in docs:
      X_unseen.append(self.features(query.split(), doc.split()))

    X_unseen = np.array(X_unseen)
    return X_unseen
  
  def get_scores(self, query, docs):
    X_unseen = self.predict_model(query, docs)
    scores = self.model_ranker.predict(X_unseen)
    print(f"The model scores: {scores}")

    did_scores = [x for x in zip([did for (did, _) in docs], scores)]
    sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

    print("query        :", query)
    print("SERP/Ranking :")
    for (did, score) in sorted_did_scores:
      print(did, score)

if __name__ == '__main__':
  query = "how much cancer risk can be avoided through lifestyle change ?"
  docs = [
    ("D1", "dietary restriction reduces insulin-like growth factor levels modulates apoptosis cell proliferation tumor progression num defici pubmed ncbi abstract diet contributes one-third cancer deaths western world factors diet influence cancer elucidated reduction caloric intake dramatically slows cancer progression rodents major contribution dietary effects cancer insulin-like growth factor igf-i lowered dietary restriction dr humans rats igf-i modulates cell proliferation apoptosis tumorigenesis mechanisms protective effects dr depend reduction multifaceted growth factor test hypothesis igf-i restored dr ascertain lowering igf-i central slowing bladder cancer progression dr heterozygous num deficient mice received bladder carcinogen p-cresidine induce preneoplasia confirmation bladder urothelial preneoplasia mice divided groups ad libitum num dr num dr igf-i igf-i/dr serum igf-i lowered num dr completely restored igf-i/dr-treated mice recombinant igf-i administered osmotic minipumps tumor progression decreased dr restoration igf-i serum levels dr-treated mice increased stage cancers igf-i modulated tumor progression independent body weight rates apoptosis preneoplastic lesions num times higher dr-treated mice compared igf/dr ad libitum-treated mice administration igf-i dr-treated mice stimulated cell proliferation num fold hyperplastic foci conclusion dr lowered igf-i levels favoring apoptosis cell proliferation ultimately slowing tumor progression mechanistic study demonstrating igf-i supplementation abrogates protective effect dr neoplastic progression"),    
    ("D2", "study hard as your blood boils"), 
    ("D3", "processed meats risk childhood leukemia california usa pubmed ncbi abstract relation intake food items thought precursors inhibitors n-nitroso compounds noc risk leukemia investigated case-control study children birth age num years los angeles county california united states cases ascertained population-based tumor registry num num controls drawn friends random-digit dialing interviews obtained num cases num controls food items principal interest breakfast meats bacon sausage ham luncheon meats salami pastrami lunch meat corned beef bologna hot dogs oranges orange juice grapefruit grapefruit juice asked intake apples apple juice regular charcoal broiled meats milk coffee coke cola drinks usual consumption frequencies determined parents child risks adjusted risk factors persistent significant associations children's intake hot dogs odds ratio num num percent confidence interval ci num num num hot dogs month trend num fathers intake hot dogs num ci num num highest intake category trend num evidence fruit intake provided protection results compatible experimental animal literature hypothesis human noc intake leukemia risk potential biases data study hypothesis focused comprehensive epidemiologic studies warranted"), 
    ("D4", "long-term effects calorie protein restriction serum igf num igfbp num concentration humans summary reduced function mutations insulin/igf-i signaling pathway increase maximal lifespan health span species calorie restriction cr decreases serum igf num concentration num protects cancer slows aging rodents long-term effects cr adequate nutrition circulating igf num levels humans unknown report data long-term cr studies num num years showing severe cr malnutrition change igf num igf num igfbp num ratio levels humans contrast total free igf num concentrations significantly lower moderately protein-restricted individuals reducing protein intake average num kg num body weight day num kg num body weight day num weeks volunteers practicing cr resulted reduction serum igf num num ng ml num num ng ml num findings demonstrate unlike rodents long-term severe cr reduce serum igf num concentration igf num igfbp num ratio humans addition data provide evidence protein intake key determinant circulating igf num levels humans suggest reduced protein intake important component anticancer anti-aging dietary interventions"), 
    ("D5", "cancer preventable disease requires major lifestyle abstract year num million americans num million people worldwide expected diagnosed cancer disease commonly believed preventable num num cancer cases attributed genetic defects remaining num num roots environment lifestyle lifestyle factors include cigarette smoking diet fried foods red meat alcohol sun exposure environmental pollutants infections stress obesity physical inactivity evidence cancer-related deaths num num due tobacco num num linked diet num num due infections remaining percentage due factors radiation stress physical activity environmental pollutants cancer prevention requires smoking cessation increased ingestion fruits vegetables moderate alcohol caloric restriction exercise avoidance direct exposure sunlight minimal meat consumption grains vaccinations regular check-ups review present evidence inflammation link agents/factors cancer agents prevent addition provide evidence cancer preventable disease requires major lifestyle")
  ]

  model = LSIAModel()
  model.train_the_model()
  model.get_scores(query, docs)
