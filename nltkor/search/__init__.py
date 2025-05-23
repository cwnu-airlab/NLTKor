# The following trick allows us to import the classes directly from the search module:
from .classical import (
    SearchAlgorithm,
    NaiveSearch,
    RabinKarpSearch,
    KMPSearch,
    BoyerMooreSearch,
)
from .faiss_search import FaissSearch
from .kobert_tokenizer import KoBERTTokenizer
from .trie_search import TRIESearch
