# -*- coding: utf-8 -*-

"""
Utility functions
"""

import re
import os, sys
import logging
import nltk
import nltkor
#from nltkor import Kor_char
from nltkor.tokenize import Ko_tokenize
import numpy as np

#from nltk.tokenize.regexp import RegexpTokenizer
#from nltk.tokenize import TreebankWordTokenizer
from . import attributes


def get_word_from_morph_lexicon(root, word, tags, space_flag):
	'''
	space_flag: if True then including space, otherwise do not including space
	'''

	values = list()
	value_data = list()
	if not word: return root.keys()

	current_dict = root
	_end = '$$'
	s = 0
	for i, letter in enumerate(word):
		#print(i, '>', letter, current_dict)
		if letter in current_dict:
			#print(letter, current_dict[letter])
			current_dict = current_dict[letter]
			if _end in current_dict :
				for idx in range(i-s): values.pop()
				values.append(word[s:i+1])
				for idx in range(i-s): value_data.pop()
				value_data.append(current_dict[_end])
			else: values.append(letter); value_data.append(tags[i])
		else:
			#print('==', letter, values)
			if space_flag or letter != ' ':
				values.append(letter)				# 최장일치 : -1
				value_data.append(tags[i])
			s = i+1
			current_dict = root
	else:
		if values: return values, value_data
		else:	return list(word), tags

def intersperse(lst, item):
	result = [item] * (len(lst) * 2 - 1)
	result[0::2] = lst
	return result

def get_word(root, word, tags, space_flag=False) :
	'''
	space_flag : True : 공백이 있어도 매칭됨
							 False: 공백이 있으면 매칭 안됨
	'''
	word_list = get_word_from_morph_lexicon(root, word, tags, space_flag)
	return word_list



def tokenize(text, use_sent_tokenizer=True):
		"""
		Call the tokenizer function for the given language.
		The returned tokens are in a list of lists, one for each sentence.

		:param use_sent_tokenizer: True : use sentence tokenizer
									False : sentence per line
		"""
		return tokenize_ko(text, use_sent_tokenizer)

def tokenize_ko(text, use_sent_tokenizer=True, clean=True):
		"""
		text: string
		Return a list of lists of the tokens in text, separated by sentences.
		"""
		if clean:
			text = clean_kotext(text)

		if use_sent_tokenizer:
			## False: 띄어쓰기 무시, True: 띄어쓰기 고려
			sentences = [Ko_tokenize.syllable(sentence, True) for sentence in Ko_tokenize.sentence(text)]
		else:
			sentences = [Ko_tokenize.syllable(text, True)]

		return sentences

def clean_kotext(text, correct=False):
		"""
		1. 특수 공백문자를 공백으로 처리
		Apply some transformations to the text, such as
		replacing digits for 9 and simplifying quotation marks.

		:param correct: If True, tries to correct punctuation misspellings.
		"""
		# replaces different kinds of quotation marks with "
		# take care not to remove apostrophes
		'''
		text = re.sub(r"(?u)(^|\W)[‘’′`']", r'\1"', text)
		text = re.sub(r"(?u)[‘’`′'](\W|$)", r'"\1', text)
		text = re.sub(r'(?u)[«»“”]', '"', text)

		if correct:
				# tries to fix mistyped tokens (common in Wikipedia-pt) as ,, '' ..
				text = re.sub(r'(?<!\.)\.\.(?!\.)', '.', text) # take care with ellipses
				text = re.sub(r'([,";:])\1,', r'\1', text)

				# inserts space after leading hyphen. It happens sometimes in cases like:
				# blablabla -that is, bloblobloblo
				text = re.sub(' -(?=[^\W\d_])', ' - ', text)
		'''

		# replaces numbers with the 9's
		text = re.sub(r'\xa0', ' ', text)
		text = re.sub(u'　', ' ', text)
		text = re.sub(u' ', ' ', text)
		text = re.sub(u' +', ' ', text) # 연속된 공백 처리
		# replaces numbers with the 9's, 사전쪽과 같이 판단할 것
		#text = re.sub(r'\d', '9', text)
		# replaces english character with the a's
		#text = re.sub(r'[a-zA-Z]', 'a', text)
		# replaces chinses character with the 家's
		#for x in re.findall(r'[\u4e00-\u9fff]', text):
		#	text = re.sub(x, '家', text)

		# replaces special ellipsis character
		#text = text.replace(u'…', '...')

		return text



def generate_feature_vectors(num_vectors, num_features, min_value=-0.1, max_value=0.1):
		"""
		Generates vectors of real numbers, to be used as word features.
		Vectors are initialized randomly. Returns a 2-dim numpy array.
		"""
		logger = logging.getLogger("Logger")
		#table = (max_value * 2) * np.random.random((num_vectors, num_vectors, num_features, num_features)) + min_value
		table = (max_value * 2) * np.random.random((num_vectors, num_features)) + min_value
		logger.debug("Generated %d feature vectors with %d features each." % (num_vectors, num_features))
		print("Generated %d feature vectors with %d features each." % (num_vectors, num_features))

		return table


def count_lines(filename):
		"""Counts and returns how many non empty lines in a file there are."""
		with open(filename, 'r') as f:
				lines = [x for x in list(f) if x.strip()]
		return len(lines)

def _create_affix_tables(affix, table_list, num_features):
		"""
		Internal helper function for loading suffix or prefix feature tables
		into the given list.
		affix should be either 'suffix' or 'prefix'.
		"""
		logger = logging.getLogger('Logger')
		logger.info('Generating %s features...' % affix)
		tensor = []
		codes = getattr(attributes.Affix, '%s_codes' % affix)
		num_affixes_per_size = getattr(attributes.Affix, 'num_%ses_per_size' % affix)
		for size in codes:

				# use num_*_per_size because it accounts for special suffix codes
				num_affixes = num_affixes_per_size[size]
				table = generate_feature_vectors(num_affixes, num_features)
				tensor.append(table)

		# affix attribute actually has a 3-dim tensor
		# (concatenation of 2d tables, one for each suffix size)
		for table in tensor:
				table_list.append(table)

def create_feature_tables(args, md, text_reader):
		"""
		Create the feature tables to be used by the network. If the args object
		contains the load_features option as true, the feature table for word types
		is loaded instead of being created. The actual number of
		feature tables will depend on the argument options.

		:param arguments: Parameters supplied to the program
		:param md: metadata about the network
		:param text_reader: The TextReader being used.
		:returns: all the feature tables to be used
		"""

		logger = logging.getLogger("Logger")
		feature_tables = []

		if not args.load_types:
				logger.info("Generating word type features...")
				table_size = len(text_reader.word_dict)
				types_table = generate_feature_vectors(table_size, args.num_features)
		else:
				logger.info("Loading word type features...")
				# check if there is a word feature file specific for the task
				# if not, load a generic one
				filename = md.paths[md.type_features]
				if os.path.exists(filename):
						types_table = load_features_from_file(filename)
				else:
						filename = md.paths['type_features']
						types_table = load_features_from_file(filename)

				if len(types_table) < len(text_reader.word_dict):
						# the type dictionary provided has more types than
						# the number of feature vectors. So, let's generate
						# feature vectors for the new types by replicating the vector
						# associated with the RARE word
						diff = len(text_reader.word_dict) - len(types_table)
						logger.warning("Number of types in feature table and dictionary differ.")
						logger.warning("Generating features for %d new types." % diff)
						num_features = len(types_table[0])
						new_vecs =	generate_feature_vectors(diff, num_features)
						types_table = np.append(types_table, new_vecs, axis=0)

				elif len(types_table) < len(text_reader.word_dict):
						logger.warning("Number of features provided is greater than the number of tokens\
						in the dictionary. The extra features will be ignored.")

		feature_tables.append(types_table)	# head
		#print(md.task)
		#if md.task in ['labeled_dependency', 'unlabeled_dependency']:
		#	feature_tables.append(types_table)	# tail

		# Capitalization
		if md.use_caps:
				logger.info("Generating capitalization features...")
				caps_table = generate_feature_vectors(attributes.Caps.num_values, args.caps)
				feature_tables.append(caps_table)

		# Prefixes
		if md.use_prefix:
				_create_affix_tables('prefix', feature_tables, args.prefix)

		# Suffixes
		if md.use_suffix:
				_create_affix_tables('suffix', feature_tables, args.suffix)

		# POS tags
		if md.use_pos:
				logger.info("Generating POS features...")
				num_pos_tags = text_reader.get_num_pos_tags()
				pos_table = generate_feature_vectors(num_pos_tags, args.pos)
				#feature_tables.append(pos_table) # head    # 여기와 *_reader의 converter와 일치해야 한다.
				feature_tables.append(pos_table) # tail

		# chunk tags
		if md.use_chunk:
				logger.info("Generating chunk features...")
				num_chunk_tags = count_lines(md.paths['chunk_tags'])
				chunk_table = generate_feature_vectors(num_chunk_tags, args.chunk)
				feature_tables.append(chunk_table)

		#print(len(feature_tables))
		return feature_tables



def set_distance_features(max_dist=None,
													num_target_features=None, num_pred_features=None):
		"""
		Returns the distance feature tables to be used by a convolutional network.
		One table is for relative distance to the target predicate, the other
		to the predicate.

		:param max_dist: maximum distance to be used in new vectors.
		"""
		logger = logging.getLogger("Logger")

		# max_dist before/after, 0 distance, and distances above the max
		max_dist = 2 * (max_dist + 1) + 1
		logger.info("Generating target word distance features...")
		target_dist = generate_feature_vectors(max_dist, num_target_features)
		logger.info("Generating predicate distance features...")
		pred_dist = generate_feature_vectors(max_dist, num_pred_features)

		return [target_dist, pred_dist]


def set_logger(level):
		"""Sets the logger to be used throughout the system."""
		log_format = '%(message)s'
		logging.basicConfig(format=log_format)
		logger = logging.getLogger("Logger")
		logger.setLevel(level)

def load_features_from_file(features_file):
		"""Reads a file with features written as binary data."""
		return np.load(features_file)

def save_features_to_file(table, features_file):
		"""Saves a feature table to a given file, writing binary data."""
		np.save(features_file, table)

def convert_iobes_to_bracket(tag):
		"""
		Convert tags from the IOBES scheme to the CoNLL bracketing.

		Example:
		B-A0 -> (A0*
		I-A0 -> *
		E-A0 -> *)
		S-A1 -> (A1*)
		O		-> *
		"""
		if tag.startswith('I') or tag.startswith('O'):
				return '*'
		if tag.startswith('B'):
				return '(%s*' % tag[2:]
		if tag.startswith('E'):
				return '*)'
		if tag.startswith('S'):
				return '(%s*)' % tag[2:]
		else:
				raise ValueError("Unknown tag: %s" % tag)

def boundaries_to_arg_limits(boundaries):
		"""
		Converts a sequence of IOBES tags delimiting arguments to an array
		of argument boundaries, used by the network.
		"""
		limits = []
		start = None

		for i, tag in enumerate(boundaries):
				if tag == 'S':
						limits.append([i, i])
				elif tag == 'B':
						start = i
				elif tag == 'E':
						limits.append([start, i])

		return np.array(limits, np.int)
