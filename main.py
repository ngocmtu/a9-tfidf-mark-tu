# Mark Raddell
import math, os, re
import numpy as np
from typing import Tuple, List, Dict


class Document:
    """The Document class.
        Attributes: text - the text of the document
                    terms - a dictionary mapping words to the number of times they occur in the document
                         - please note that this sort of dictionary is returned by tokenize
                    term_vector - an ordered list of tfidf values for each term. The length of this list
                                  will be the same for all documents in a corpus. 
    """ 
    def __init__(self):
        """Creates an empty document. 
        """
        self.text = ""
        self.terms = {}
        self.term_vector = []

    def __str__(self):
        """Returns the first 500 characters in the document as a preview. 
        """
        return self.text[:500]

class TFIDF_Engine:
    """The TFIDF_Engine class. 
        Attributes: corpus_location - a relative path to the folder of documents in the corpus
                    documents - a list of document objects, initially empty
                    N - the number of documents in the corpus
                    df_table - a dictionary mapping words to the number of documents they occurred in
                    term_vector_words - an ordered list of the unique words in the corpus. This list
                                will dictate the order of the scores in each document.term_vector
    """

    def __init__(self):
        self.corpus_location = "news_corpus"
        self.documents = []
        self.N = 0
        self.df_table = {}
        self.term_vector_words = []

    def __str__(self) :
        s = "corpus has: " + str(self.N) + " documents\n"
        s += "beginning of doc vector words is: \n" + str(self.term_vector_words[:25])
        return s

    def read_files(self):
        """Gathers files in folder, self.corpus_location. For each file, reads in content and creates a
            document object, sets the text (the raw text) and terms (the result of tokenize, a map of words
            to counts) attributes, appends each document object to self.documents. Sets self.N to the number
            of documents that it read in. 
        """
        #files_in_folder = list()
        #for file_name in os.listdir(self.corpus_location):
            #files_in_folder.append(os.path.join(self.corpus_location, file_name))

        files_in_folder = [os.path.join(self.corpus_location, file_name) for file_name in os.listdir(self.corpus_location)]

        for f_path in files_in_folder:
            document_name = Document()
            with open(f_path, 'r') as file:
                file_text = file.read()
                document_name.text = file_text
                file_tokens = self.tokenize(file_text)
                for token in file_tokens:
                    if token not in document_name.terms:
                        document_name.terms[token] = 1
                    else:
                        document_name.terms.update(token,document_name.terms[token] + 1)
                self.documents.append(document_name)
            self.N += 1


    def create_df_table(self):
        """Iterates over the document objects in self.documents and generates self.df_table,
            mapping all words in the corpus to the number of documents they occur in. Utilizes the
            terms attribtue of each document in constructing self.df_table, rather than tokenizing again.
            Creates self.term_vector_words which holds the order of words for the document
            vector, to be used later. Any order is fine, but once set, should not be changed.
        """
        table_of_words = dict()
        for doc in self.documents:
            for word in doc.terms:
                if word not in table_of_words:
                    table_of_words[word] = 1
                    self.term_vector_words.append(word) #makes sure it matches order of table of words
                elif word in table_of_words:
                    table_of_words[word] += 1
        self.df_table.update(table_of_words)

    def create_term_vector(self, d: Document):
        """Creates a term vector for document d, storing it in the 'term_vector' attribute
            of document d. Must handle situations where a word in d (if d is a query) is not
            in the corpus. For example, if we search for "cheese pizza" and "cheese" is not
            in the corpus. We should simply skip query terms that do not occur in the corpus.
            Remember that the order of the document.term_vector is determined by self.term_vector_words.

            Args: a document, d - this could be a document from the corpus or a document representing a query
        """

        total_num_words = 0
        for term, number_occurred in d.terms.items():
            total_num_words += number_occurred

        for word in self.term_vector_words:
            tfidf = 0
            for term, number_occurred in d.terms.items():
                if term == word:
                    tf = number_occurred / total_num_words
                    documents_with_term = self.df_table[term]
                    idf = math.log(len(self.documents) / documents_with_term)
                    tfidf = tf * idf

            d.term_vector.append(tfidf)
        pass 

    def create_term_vectors(self):
        """Creates a term_vector for each document, utilizing self.create_term_vector.
        """
        for document in self.documents:
            self.create_term_vector(document)
        pass

    def calculate_cosine_sim(self, d1: Document, d2: Document) -> float:
        """Calculates the cosine simularity between two documents, the dot product of their
            term vectors.

            Args:
                two documents, d1 and d2

            Returns:
                the dot product of the term vectors of the input documents
        """
        cosine_similarity = np.dot(d1.term_vector, d2.term_vector)
        """dot_product = sum(d1.term_vector[term] * d2.term_vector[term] for term in self.term_vector_words)
        magnitude_d1 = math.sqrt(sum(float(value) ** 2 for value in d1.term_vector))
        magnitude_d2 = math.sqrt(sum(float(value) ** 2 for value in d2.term_vector))

        if magnitude_d1 == 0 or magnitude_d2 == 0:
            return 0.0  # Avoid division by zero

        cosine_similarity = dot_product / (magnitude_d1 * magnitude_d2)"""
        return cosine_similarity


    def get_results(self, query: str) -> List[Tuple[float, int]]:
        """Transforms the input query into a document (with text, terms and term_vector attributes).
            Generates similarity scores between the query doc and all other documents in the corpus,
            self.documents. The index of the document in self.documents will serve as an indentifier
            for the search result.
            Returns a list of tuples of (similarity score, index) where index refers to the position of
            the document in self.documents. This list of tuples should be sorted in descending order by
            similarity score, that is, the highest similarity score adn corresponding index will be
            first in the list.
        """
        query_document = Document()
        query_document.text = query
        file_tokens = self.tokenize(query_document.text)
        for token in file_tokens:
            if token not in query_document.terms:
                query_document.terms[token] = 1
            elif token in query_document.terms:
                query_document.terms[token] += 1

        self.create_term_vector(query_document)

        similarity_scores = list()
        doc_count = 0
        for document in self.documents:
            doc_count += 1
            similarity_scores.append(tuple(self.calculate_cosine_sim(query_document, document), doc_count))

        similarity_scores.sort(reverse=True, key=lambda x: x[0])
        return similarity_scores
                

    def query_loop(self):
        """Asks the user for a query. Utilizes self.get_results. Prints the top 5 results.  
        """
        
        print("Welcome!\n")
        while True:
            try:
                print()
                query = input("Query: ")
                sim_scores = self.get_results(query)
                #display the top 5 results
                print("RESULTS\n")
                for i in range(5):
                    print("\nresult number " + str(i) + " has score " + str(sim_scores[i][0]))
                    print(self.documents[sim_scores[i][1]])

            except (KeyboardInterrupt, EOFError):
                break

        print("\nSo long!\n")

    def tokenize(self, text: str) -> Dict[str, int]:
        """Splits given text into a list of the individual tokens and counts them

        Args:
            text - text to tokenize

        Returns:
            a dictionary mapping tokens from the input text to the number of times
            they occurred
        """
        tokens = []
        token = ""
        for c in text:
            if (
                re.match("[a-zA-Z0-9]", str(c)) != None
                or c == "'"
                or c == "_"
                or c == "-"
            ):
                token += c
            else:
                if token != "":
                    tokens.append(token.lower())
                    token = ""
                if c.strip() != "":
                    tokens.append(str(c.strip()))

        if token != "":
            tokens.append(token.lower())

        #make a dictionary mapping tokens to counts
        d_tokens = {}
        for t in tokens:
            if t in d_tokens:
                d_tokens[t] += 1
            else:
                d_tokens[t] = 1

        return d_tokens

# Optional Challenge: 
# Different to previous assignments, the test cases contained here will likely not show up
# on pycharm where you can directly see how many of them passes or fails. 
# Based on the pytest documentation and your experience with previous assignments, see
# if you can create a tfidf_test.py by converting the test cases below to the pytest format.
# Pytest documentation : https://docs.pytest.org/en/7.1.x/getting-started.html

if __name__ == "__main__":
    
    t = TFIDF_Engine()
    
    # read the files , populating self.documents and self.N
    t.read_files()
    # tests
    assert t.N == 122, "read files N test"
    assert t.documents[5].text != "", "read files document text test"
    assert t.documents[100].terms != {}, "read files document terms test"
    assert isinstance(t.documents[9].terms["the"], int), "read files document terms structure test"

    # create self.df_table from the documents
    t.create_df_table()
    
    assert t.df_table["the"] == 122, "df_table 'the' count test"
    assert t.df_table["star"] == 102, "df_table 'star' count test"
    assert 11349 <= len(t.df_table) <= 11352, "df_table number of unique words test"

    #create the document vector for each document
    t.create_term_vectors()

    assert len(t.documents[10].term_vector) == len(t.term_vector_words), "create_term_vectors test"

    #tests for calculate_cosine_sim
    assert t.calculate_cosine_sim(t.documents[0], t.documents[1]) > 0, "calculate_cosine_sim test 1"
    assert t.calculate_cosine_sim(t.documents[0], t.documents[1]) < 1, "calculate_cosine_sim test 1"
    assert abs(t.calculate_cosine_sim(t.documents[0], t.documents[0]) - 1) < 0.01

    
    #tests for get_results
    # assert t.get_results("star wars")[0][1] == 111, "get_results test 1"
    assert "Lucas announces new 'Star Wars' title" in t.documents[t.get_results("star wars")[0][1]].text, "get_results test 1"
    # assert t.get_results("movie trek george lucas")[2][1] == 24, "get_results test 2"
    assert "Stars of 'X-Men' film are hyped, happy, as comic heroes" in t.documents[t.get_results("movie trek george lucas")[2][1]].text 
    assert len(t.get_results("star trek")) == len(t.documents), "get_results test 3"

    # t.query_loop() #uncomment this line to try out the search engine

    """read in documents, create df table, create term vector - go through corpus three times 
    - this 3 step process is called indexing
    
    last - create cosign similarity between documents"""

    
