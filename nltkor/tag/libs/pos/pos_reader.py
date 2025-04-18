# -*- coding: utf-8 -*-

"""
Class for dealing with POS data.
"""

import chardet
from ..reader import TaggerReader

class ConllPos(object):
	"""
	Dummy class for storing column positions in a conll file.
	"""
	id = 0
	word = 1
	pos = 2
	SEP = '\t'

class POSReader(TaggerReader):
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
		
		super(POSReader, self).__init__(md, load_dictionaries=load_dictionaries)
		
	@property
	def task(self):
		"""
		Abstract Base Class (ABC) attribute.
		"""
		return 'pos'
	
	def _read_plain(self, filename):
		"""
		Read data from a "plain" file, with one sentence per line, each token
		as token_tag.
		"""
		self.sentences = []
		with open(filename, 'rb') as f: 
			raw_data = f.read(1024)
			detected = chardet.detect(raw_data).get('encoding', 'utf-8')
		with open(filename, 'rt', encoding = detected) as f:
			for line in f:
				#line = unicode(line, 'utf-8')
				items = line.split(ConllPos.SEP)
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
		with open(filename, 'rb') as f: 
			raw_data = f.read(1024)
			detected = chardet.detect(raw_data).get('encoding', 'utf-8')
		with open(filename, 'rt', encoding = detected) as f:
			for line in f:
				line = line.strip()
				if line.strip() == '':
					if len(sentence) > 0:
						self.sentences.append(sentence)
						sentence = []
						continue
				
				#line = unicode(line, 'utf-8')
				fields = line.split(ConllPos.SEP)
				word = fields[ConllPos.word]
				pos = fields[ConllPos.pos]
				sentence.append((word, pos))
		
		if len(sentence) > 0:
			self.sentences.append(sentence)


# backwards compatibility
MacMorphoReader = POSReader
