# -*- coding: utf-8 -*-
# Natural Language Toolkit: Taggers
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
"""
NLTK Taggers

This package contains classes and interfaces for part-of-speech
tagging, or simply "tagging".

A "tag" is a case-sensitive string that specifies some property of a token,
such as its part of speech.  Tagged tokens are encoded as tuples
``(tag, token)``.  For example, the following tagged token combines
the word ``'fly'`` with a noun part of speech tag (``'NN'``):

    >>> tagged_tok = ('fly', 'NN')

An off-the-shelf tagger is available for English. It uses the Penn Treebank tagset:

    >>> from nltk import pos_tag, word_tokenize
    >>> pos_tag(word_tokenize("John's big idea isn't all that bad."))
    [('John', 'NNP'), ("'s", 'POS'), ('big', 'JJ'), ('idea', 'NN'), ('is', 'VBZ'),
    ("n't", 'RB'), ('all', 'PDT'), ('that', 'DT'), ('bad', 'JJ'), ('.', '.')]

A Russian tagger is also available if you specify lang="rus". It uses
the Russian National Corpus tagset:

    >>> pos_tag(word_tokenize("Илья оторопел и дважды перечитал бумажку."), lang='rus')    # doctest: +SKIP
    [('Илья', 'S'), ('оторопел', 'V'), ('и', 'CONJ'), ('дважды', 'ADV'), ('перечитал', 'V'),
    ('бумажку', 'S'), ('.', 'NONLEX')]

This package defines several taggers, which take a list of tokens,
assign a tag to each one, and return the resulting list of tagged tokens.
Most of the taggers are built automatically based on a training corpus.
For example, the unigram tagger tags each word *w* by checking what
the most frequent tag for *w* was in a training corpus:

    >>> from nltk.corpus import brown
    >>> from nltk.tag import UnigramTagger
    >>> tagger = UnigramTagger(brown.tagged_sents(categories='news')[:500])
    >>> sent = ['Mitchell', 'decried', 'the', 'high', 'rate', 'of', 'unemployment']
    >>> for word, tag in tagger.tag(sent):
    ...     print(word, '->', tag)
    Mitchell -> NP
    decried -> None
    the -> AT
    high -> JJ
    rate -> NN
    of -> IN
    unemployment -> None

Note that words that the tagger has not seen during training receive a tag
of ``None``.

We evaluate a tagger on data that was not seen during training:

    >>> tagger.evaluate(brown.tagged_sents(categories='news')[500:600])
    0.73...

For more information, please consult chapter 5 of the NLTK Book.
"""


from nltkor.tag.espresso_tag import EspressoTagger
#import nltkor.tag
from nltkor.tag.libs import taggers
