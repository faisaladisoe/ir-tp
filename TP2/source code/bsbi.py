import os
import math
import time
import heapq
import pickle
import contextlib

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from nltk.tokenize import sent_tokenize, word_tokenize
from index import InvertedIndexReader, InvertedIndexWriter

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.tdtf_combination = []

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        folder_path = f'./collection/{block_dir_relative}'
        files = os.listdir(folder_path)
        paths = []

        # sentence segmentation & tokenisation
        tokenize_result_with_punct = []
        for file in files:
            path = f'{folder_path}/{file}'
            paths.append(path)
            with open(path, 'r') as f:
                tokenize_result_with_punct.append([word_tokenize(t) for t in sent_tokenize(f.read())])
        
        tokenize_result_wo_punct = []
        for file in tokenize_result_with_punct:
            same_file_result = []
            for sentence in file:
                same_file_result.append([word for word in sentence if word.isalnum()])
            tokenize_result_wo_punct.append(same_file_result)
        
        # stemming
        stemmed_result_in_general = []
        stemmer = SnowballStemmer('english')
        for file in tokenize_result_wo_punct:
            same_file_result = []
            for sentence in file:
                same_file_result.append([stemmer.stem(word) for word in sentence])
            stemmed_result_in_general.append(same_file_result)

        # stopwords removal
        stopword_removed_result_in_general = []
        stop_word_remover = set(stopwords.words('english'))
        for file in stemmed_result_in_general:
            same_file_result = []
            for sentence in file:
                same_file_result.append([word for word in sentence if not word.lower() in stop_word_remover])
            stopword_removed_result_in_general.append(same_file_result)
        
        stopword_removed_result_in_sentences = []
        for file in stopword_removed_result_in_general:
            whole_sentence = ''
            for sentence in file:
                whole_sentence += ' '.join(sentence) + ' '
            stopword_removed_result_in_sentences.append(whole_sentence.strip())
        
        # term and doc mapping
        term_id_map_result = []
        words_per_file = [file.split(' ') for file in stopword_removed_result_in_sentences]
        for sentence in words_per_file:
            for term in sentence:
                term_id_map_result.append(self.term_id_map[term])
        doc_id_map_result = [self.doc_id_map[docname] for docname in paths]

        # pair matching
        td_pairs = []
        tdtf_pairs = []
        for idx in range(len(words_per_file)):
            for term in words_per_file[idx]:
                termId = self.term_id_map[term]
                docname = paths[idx]
                docId = self.doc_id_map[docname]
                tfTerm = words_per_file[idx].count(term)
                td_pairs.append((termId, docId))
                tdtf_pairs.append((termId, docId, tfTerm))

        self.tdtf_combination = tdtf_pairs
        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        for term_id, doc_id, tf_term in self.tdtf_combination:
            if term_id not in term_dict:
                term_dict[term_id] = set()
            term_dict[term_id].add((doc_id, tf_term))
        for term_id in sorted(term_dict.keys()):
            term_value = sorted(list(term_dict[term_id]), key=lambda x: x[0])
            postings_list = []
            tf_list = []
            for doc_id, tf_term in term_value:
                postings_list.append(doc_id)
                tf_list.append(tf_term)
            if len(postings_list) == len(tf_list):
                index.append(term_id, postings_list, tf_list)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        self.load()

        # stemming
        stemmer = SnowballStemmer('english')
        stemmed_query = [stemmer.stem(word) for word in word_tokenize(query)]

        # stopwords removal
        stop_word_remover = set(stopwords.words('english'))
        stopword_removed_query = [word for word in stemmed_query if not word.lower() in stop_word_remover]

        term_id_of_query = [self.term_id_map[word] for word in stopword_removed_query]

        data_of_each_query = []
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as final_index:
            for item in final_index:
                if item[0] in term_id_of_query:
                    data_of_each_query.append(item)
        
        merge_pl_and_tfl = []
        for token_data in data_of_each_query:
            pl_and_tfl = []
            for idx in range(len(token_data[1])):
                pl_and_tfl.append((token_data[1][idx], token_data[2][idx]))
            merge_pl_and_tfl.append((token_data[0], pl_and_tfl))
        
        merge_all_pl_and_tfl_in_query = []
        for _, pl_and_tfl in merge_pl_and_tfl:
            merge_all_pl_and_tfl_in_query = sorted_merge_posts_and_tfs(merge_all_pl_and_tfl_in_query, pl_and_tfl)

        mapping_term_id_with_its_data = {}
        for term_id, pl, tf in data_of_each_query:
            mapping_term_id_with_its_data[term_id] = (pl, tf)

        heap_scores = []
        N = len(self.doc_id_map)
        for doc_id, term_freq in merge_all_pl_and_tfl_in_query:
            for term in term_id_of_query:
                if doc_id in mapping_term_id_with_its_data[term][0]:
                    wtD = (1 + math.log10(term_freq))
                    wtQ = math.log10(N // len(mapping_term_id_with_its_data[term][0]))
                    score = wtD * wtQ
                    heapq.heappush(heap_scores, (score, self.doc_id_map[doc_id]))
        
        return heapq.nlargest(k, heap_scores, key=lambda x: x[0])

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
