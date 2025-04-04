# -*- coding: utf-8 -*-

"""
Class for dealing with SRL data.
"""

from collections import defaultdict
import _pickle as cPickle
import _pickle
import logging
import re
import os
import numpy as np
#from itertools import izip

from .. import attributes
from .. import utils
from ..word_dictionary import WordDictionary
from .. import reader

class ConllPos(object):
	"""
	Dummy class for storing the position of each field in a
	CoNLL data file.
	"""
	id = 0
	word = 1
	lemma = 2
	hmorph = 3		# head morph
	hpos = 4			# head pos
	tmorph = 5		# tail morph
	tpos = 6			# tail pos
	parse = 7
	rel = 8
	semantic_role = 9
	SEP = '\t'

class SRLReader(reader.TaggerReader):
	
	def __init__(self, md=None, filename=None):
		"""
		The reader will read sentences from a given file. This file must
		be in the correct format (one token per line, columns indicating
		which tokens are predicates and their argument structure).
		
		:param filename: a file with CoNLL-like format data. If it is None,
			the reader will be created with no data.
		:param only_boundaries: train to identify only argument boundaries
		:param only_classify: train to classify pre-determined argument
		:param only_predicates: train to identify only predicates
		"""
		
		self.taskname = 'srl'
		self.pos_dict = {}
		
		if filename is not None:
			self._read_conll(filename)
			#self._clean_text()
		
		super(SRLReader, self).__init__(md)
		
	
	@property
	def task(self):
		"""
		Abstract Base Class (ABC) attribute.
		"""
		return self.taskname
	
	
	def _read_conll(self, filename):
		'''
		Read a file in CoNLL format and extracts semantic role tags
		for each token.
		'''
		lines = []
		with open(filename, 'rt') as f:
			for line in f:
				line = line.strip()
				lines.append(line)
		
		self.sentences = []
		self.predicates = []
		tokens = []
		sent_predicates = []
		sent_tags = []
		token_number = 0
		
		for line in lines:
			line = line.strip()
			
			if line == '':
				# blank line between sentences
				if len(tokens) > 0:
					sentence = (tokens, sent_tags)
					self.sentences.append(sentence)
					self.predicates.append(np.array(sent_predicates))
					tokens = []
					sent_predicates = []
					sent_tags = []
					token_number = 0
				
				continue
			
			fields = line.split(ConllPos.SEP)
			idx = fields[ConllPos.id]
			word = fields[ConllPos.word]
			lemma = fields[ConllPos.lemma]
			hmorph = fields[ConllPos.hmorph]
			hpos = fields[ConllPos.hpos].lower()
			tmorph = fields[ConllPos.tmorph]
			tpos = fields[ConllPos.tpos].lower()
			parse = fields[ConllPos.parse]
			rel = fields[ConllPos.rel]
			is_predicate = (rel[:1] == 'V')
			tag = fields[ConllPos.semantic_role]
			
			tag = self._read_role(tag)	
			sent_tags.append((int(parse)-1, tag))			# note: codify_sentences

			token = attributes.Token(word=word, morph_h=hmorph, morph_t=tmorph, pos_t=tpos, chunk=rel)
			#token = attributes.Token(word, morph_h=hmorph, pos_h=hpos, morph_t=tmorph, pos_t=tpos, chunk=rel)
			tokens.append(token)
			if is_predicate:
				sent_predicates.append(token_number)
			
			token_number += 1
		
		if len(tokens) > 0:
			# last sentence
			sentence = (tokens, sent_tags)
			self.sentences.append(sentence)
			self.predicates.append(np.array(sent_predicates))
	
	@classmethod
	def _read_role(cls, role):
		'''
		Reads the semantic role from a CoNLL-style file.

		pram: role what is read from the conll file
		'''
		return role 

	def extend(self, data):
		"""
		Adds more data to the reader.
		:param data: a list of tuples in the format (tokens, tags, predicates), 
		one for each sentence.
		"""
		self.sentences.extend([(sent, tags) for sent, tags, _ in data])
		self.predicates.extend([np.array(preds) for _, _, preds in data])
	
	def load_or_create_tag_dict(self):
		"""
		In the case of SRL argument classification or one step SRL, try to 
		load the tag dictionary. If the file with the tags is not present,
		a new one is created from the available sentences. 
		
		In the case of argument detection or predicate detection, 
		this function does nothing.
		"""
		if os.path.isfile(self.md.paths['srl_tags']):
			self.load_tag_dict()
			return
		
		self._create_tag_dict()
		logger = logging.getLogger('Logger')
		logger.info('Created SRL tag dictionary')
	
	def _create_tag_dict(self):
		"""
		Examine the available sentences and create a tag dictionary.
		
		:param iob: If True, this function will generate an entry for B-[tag] 
			and one for I-[tag], except for the tag 'O'.
		"""
		logger = logging.getLogger("Logger")
		tags = {tag
				for _, tags in self.sentences
				for rel, tag in tags}
		
		# create a dictionary now even if uses IOB, in order to save it in 
		# a deterministic order
		self.tag_dict = {tag: code for code, tag in enumerate(tags)}
		reader.save_tag_dict(self.md.paths['srl_tags'], self.tag_dict)
		logger.debug("Saved SRL tag dictionary.")
		
	
	def load_tag_dict(self, filename=None, iob=False):
		"""
		Loads the tag dictionary from the default file. The dictionary file should
		have one tag per line. 
		
		:param iob: If True, this function will generate an entry for B-[tag] 
			and one for I-[tag], except for the tag 'O'.
		"""
		if filename is None:
			filename = self.md.paths['srl_tags']
		
		self.tag_dict = {}
		code = 0
		with open(filename, 'rt') as f:
			for tag in f:
				tag = tag.strip()
				if tag == '':
					continue
				
				self.tag_dict[tag] = code
				
				code += 1
		
	
	def _generate_iobes_dictionary(self):
		"""
		Generate the reader's tag dictionary mapping the IOBES tags to numeric codes.
		"""
		self.tag_dict = {tag: code for code, tag in enumerate('IOBES')}
	
	def _generate_predicate_id_dictionary(self):
		"""
		Generate a tag dictionary for identifying predicates.
		It has two tags: V for predicates and O for others.
		"""
		self.tag_dict = {'-': 0, 'V': 1}
		#self.tag_dict = {'O': 0, 'V': 1}
	
	def generate_dictionary(self, dict_size=None, minimum_occurrences=2):
		"""
		Generates a token dictionary based on the given sentences.
		
		:param dict_size: Max number of tokens to be included in the dictionary.
		:param minimum_occurrences: Minimum number of times that a token must
			appear in the text in order to be included in the dictionary.
		"""
		logger = logging.getLogger("Logger")
		all_tokens = [token.word
					  for tokens, _ in self.sentences
					  for token in tokens]
		self.word_dict = WordDictionary(all_tokens, dict_size, minimum_occurrences)
		logger.info("Created dictionary with %d tokens" % self.word_dict.num_tokens)
	
	def _clean_text(self):
		"""
		Cleans the sentences text, replacing numbers for a keyword, different
		kinds of quotation marks for a single one, etc.
		"""
		for sent, _ in self.sentences:
			for i, token in enumerate(sent):
				new_word = utils.clean_text(token.word, correct=False)
				new_lemma = utils.clean_text(token.lemma, correct=False) 
				token.word = new_word
				token.lemma = new_lemma
				sent[i] = token

	def create_converter(self):
		"""
		This function overrides the TextReader's one in order to deal with Token
		objects instead of raw strings.
		"""
		self.converter = attributes.TokenConverter()
		
		if self.md.use_lemma:
			# look up word lemmas 
			word_lookup = lambda t: self.word_dict.get(t.lemma)
		else:
			# look up the word itself
			word_lookup = lambda t: self.word_dict.get(t.word)
			 
		self.converter.add_extractor(word_lookup)
		
		#if self.md.use_caps:
		#	caps_lookup = lambda t: attributes.get_capitalization(t.word)
		#	self.converter.add_extractor(caps_lookup)
		
		if self.md.use_pos:
			with open(self.md.paths['pos_tag_dict']) as f:
				#pos_dict = cPickle.load(f)
				buf = f.readlines()
				for i, line in enumerate(buf):
					line = line.strip()
					self.pos_dict[line] = i

			pos_def_dict = defaultdict(lambda: self.pos_dict['NN'])
			pos_def_dict.update(self.pos_dict)
			pos_lookup = lambda t: pos_def_dict[t.pos_t]
			self.converter.add_extractor(pos_lookup)
		
		#if self.md.use_chunk:
		#	with open(self.md.paths['chunk_tag_dict']) as f:
		#		chunk_dict = cPickle.load(f)
			
		#	chunk_def_dict = defaultdict(lambda: chunk_dict['O'])
		#	chunk_def_dict.update(chunk_dict)
		#	chunk_lookup = lambda t: chunk_def_dict[t.chunk]
		#	self.converter.add_extractor(chunk_lookup)
	
	def get_num_pos_tags(self):
		return len(self.pos_dict)

	def generate_tag_dict(self):
		"""
		Generates a tag dictionary that converts the tag itself
		to an index to be used in the neural network.
		"""
		self.tagset = set(tag
						  for _, props in self.sentences
						  for prop in props
						  for _, tag in prop)
		
		self.tag_dict = dict( zip( self.tagset,
								   xrange(len(self.tagset))
								   )
							 )
	
	def _remove_tag_names(self):
		"""Removes the actual tag names, leaving only IOB or IOBES block delimiters."""
		for _, propositions in self.sentences:
			for tags in propositions:
				for i, (_, tag) in enumerate(tags):
					tags[i] = tag[0]
	
	def _codify_sentences(self):
		"""Internal helper function."""
		new_sentences = []
		self.tags = []
		
		for (sent, props) in self.sentences:
			new_sent = []
			sentence_tags = []
			
			for token in sent:
				new_token = self.converter.convert(token)
				new_sent.append(new_token)
			
			for prop in props:
				sentence_tags.append(prop)

			new_sentences.append(np.array(new_sent))
			self.tags.append(sentence_tags)
		#print(new_sentences, flush=True)
		#print(self.tags, flush=True)
		
		self.sentences = new_sentences
		self.codified = True
	
	def codify_sentences(self):
		"""
		Converts each token in each sequence into indices to their feature vectors
		in feature matrices. The previous sentences as text are not accessible anymore.
		Tags are also encoded. This function takes care of the case of classifying 
		pre-delimited arguments.
		"""
		if self.converter is None:
			self.create_converter()
		
		self._codify_sentences()
		self.arg_limits = []
		
		for i, propositions in enumerate(self.tags):
			new_sent_tags = []
			sent_args = []
			
			for j, (rel, prop_tags) in enumerate(propositions):
				
				new_prop_tags = []
				prop_args = []
				
				#if prop_tags != '-' and j==rel:
				if prop_tags != '-' :
					prop_args.append(np.array([j, j+1]))
					new_prop_tags.append(self.tag_dict[prop_tags])

				sent_args.append(np.array(prop_args))
				new_sent_tags.append(np.array(new_prop_tags))
			
			self.arg_limits.append(sent_args)
			self.tags[i] = new_sent_tags
					 
	
	def convert_tags(self, scheme, update_tag_dict=True, only_boundaries=False):
		"""
		Replaces each word label with an IOB or IOBES version, appending a prefix
		to them. 
		
		:param scheme: IOB or IOBES (In, Other, Begin, End, Single).
		:param update_dict: whether to update or not the tag dictionary after
			converting the tags.
		:param only_boundaries: if True, only leaves the IOBES tags and remove
			the actual tags. Also, avoid updating the tag dict.
		"""
		scheme = scheme.lower()
		if scheme not in ('iob', 'iobes'):
			raise ValueError("Unknown tagging scheme: %s" % scheme)
		
		for _, props in self.sentences:
			for prop in props:
				
				last_tag = None
				for i, tag in enumerate(prop):
					
					if tag == 'O':
						# O tag is independent from IBES
						last_tag = tag 
						continue
					
					try:
						next_tag = prop[i + 1]
					except IndexError:
						# last word already
						next_tag = None
					 
					if tag != last_tag:
						# a new block starts here. 
						last_tag = tag
						if scheme == 'iob' or next_tag == tag:
							prop[i] = 'B-%s' % tag
						else:
							prop[i] = 'S-%s' % tag
					else:
						# the block continues. 
						if scheme == 'iob' or next_tag == tag:
							prop[i] = 'I-%s' % tag
						else:
							prop[i] = 'E-%s' % tag
			
		if only_boundaries:
			self._remove_tag_names()
		elif update_tag_dict:
			self.generate_tag_dict()
		else:
			# treat any tag not appearing in the tag dictionary as O
			actual_tagset = {tag for _, props in self.sentences for prop in props for tag in prop}
			for tag in actual_tagset:
				if tag not in self.tag_dict:
					self.tag_dict[tag] = self.tag_dict[self.rare_tag]
	
