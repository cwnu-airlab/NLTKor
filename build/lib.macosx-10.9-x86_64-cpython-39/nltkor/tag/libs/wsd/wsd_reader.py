# -*- coding: utf-8 -*-

"""
Class for dealing with WSD data.
"""

from ..reader import TaggerReader

class ConllWSD(object):
	"""
	Dummy class for storing column positions in a conll file.
	"""
	id = 0
	word = 1
	pos = 2
	wsd = 3
	SEP = '\t'

class WSDReader(TaggerReader):
	"""
	This class reads data from a POS corpus and turns it into a format
	readable by the neural network for the POS tagging task.
	"""
	
	def __init__(self, md=None, filename=None, load_dictionaries=True):
		"""
		Constructor
		"""
		self.rare_tag = None
		self.sentences = []
		if filename is not None:
			try:
				self._read_plain(filename)
			except:
				self._read_conll(filename)
		
		super(WSDReader, self).__init__(md, load_dictionaries=load_dictionaries)
		
	@property
	def task(self):
		"""
		Abstract Base Class (ABC) attribute.
		"""
		return 'wsd'
	
	def _read_plain(self, filename):
		"""
		Read data from a "plain" file, with one sentence per line, each token
		as token_tag.
		"""
		self.sentences = []
		with open(filename, 'rt') as f:
			for line in f:
				#line = unicode(line, 'utf-8')
				items = line.strip().split()
				sentence = []
				for item in items:
					token, tag = item.rsplit('_', 1)
					sentence.append((token, tag))
					
				self.sentences.append(sentence)
	
	def _read_conll(self, filename):
		"""
		Read data from a CoNLL formatted file. It expects at least 4 columns:
		id, surface word, lemma (ignored, may be anything) 
		and the POS tag.
		"""
		self.sentences = []
		sentence = []
		with open(filename, 'rt') as f:
			for line in f:
				line = line.strip()
				if line == '':
					if len(sentence) > 0:
						self.sentences.append(sentence)
						sentence = []
						continue
				
				fields = line.split(ConllWSD.SEP)
				try:
					word = fields[ConllWSD.word]
					pos = fields[ConllWSD.pos]
					wsd = fields[ConllWSD.wsd]
				except: continue
				sentence.append((word, wsd))
				#sentence.append((word, pos, ner))
		
		if len(sentence) > 0:
			self.sentences.append(sentence)

# backwards compatibility
MacMorphoReader = WSDReader
