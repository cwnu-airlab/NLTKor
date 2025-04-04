from nltk.translate.bleu_score import *
from nltkor.tokenize import Ko_tokenize
import numpy as np
import torch
import time
import math

def bleu_tensor(reference,candidate,n=0, smoothing_function=None):
    if n: weights = tuple(1 if i == n-1 else 0 for i in range(4))
    else: weights = (0.25, 0.25, 0.25, 0.25)

 
  
    reference=reference.unsqueeze(1)
    reference=reference.numpy()
    candidate=candidate.numpy()
    return torch.tensor(corpus_bleu(reference,candidate,weights,smoothing_function=smoothing_function))



