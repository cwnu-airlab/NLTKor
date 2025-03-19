# -*- coding: utf-8 -*-

"""
Taggers wrapping the neural networks.
"""

import logging
#from os import major
import numpy as np
import re
from nltkor import etc

from . import utils
from . import config
from . import attributes
from .metadata import Metadata
from .pos import POSReader
from .ner import NERReader
from .wsd import WSDReader
from .srl import SRLReader
from .parse import DependencyReader
import sys
sys.path.append("libs/")
from .network import Network, ConvolutionalNetwork, ConvolutionalDependencyNetwork


def load_network(md):
		"""
		Loads the network from the default file and returns it.
		"""
		logger = logging.getLogger("Logger")
		is_srl = md.task == 'srl'

		logger.info('Loading network')
		if is_srl :
				net_class = ConvolutionalNetwork
		elif md.task.endswith('dependency'):
				net_class = ConvolutionalDependencyNetwork
		else:
				net_class = Network

		nn = net_class.load_from_file(md.paths[md.network])

		logger.info('Done')
		return nn


def create_reader(md, gold_file=None):
		"""
		Creates a TextReader object for the given task and loads its dictionary.
		:param md: a metadata object describing the task
		:param gold_file: path to a file with gold standard data, if
				the reader will be used for testing.
		"""
		logger = logging.getLogger('Logger')
		logger.info('Loading text reader...')

		if md.task == 'pos':
				tr = POSReader(md, filename=gold_file)

		elif md.task == 'ner':
				tr = NERReader(md, filename=gold_file)

		elif md.task == 'wsd':
				tr = WSDReader(md, filename=gold_file)

		elif 'dependency' in md.task:
				labeled = md.task.startswith('labeled')
				tr = DependencyReader(md, filename=gold_file, labeled=labeled)

		elif md.task.startswith('srl'):
				tr = SRLReader(md, filename=gold_file)

		else:
				raise ValueError("Unknown task: %s" % md.task)

		logger.info('Done')
		return tr

def _group_arguments(tokens, predicate_positions, arg_tokens, labels):
		"""
		Groups words pertaining to each argument and returns a dictionary for each predicate.
		"""
		print(tokens, predicate_positions, arg_tokens, labels)
		arg_structs = []

		for predicate_position, pred_arg_tokens, pred_labels in zip(predicate_positions,
																																 arg_tokens,
																																 labels):
				structure = {}

				for tag, arg_token in zip(pred_labels, pred_arg_tokens):
					#argument_tokens = [token]
					#tag = pred_labels.pop(0)
					structure[tag] = [arg_token]

				predicate = tokens[predicate_position-1]
				arg_structs.append((predicate, structure))

		return arg_structs


class SRLAnnotatedSentence(object):
		"""
		Class storing a sentence with annotated semantic roles.

		It stores a list with the sentence tokens, called `tokens`, and a list of tuples
		in the format `(predicate, arg_strucutres)`. Each `arg_structure` is a dict mapping
		semantic roles to the words that constitute it. This is used instead of a two-level
		dictionary because one sentence may have more than one occurrence of the same
		predicate.

		This class is used only for storing data.
		"""

		def __init__(self, tokens, arg_structures):
				"""
				Creates an instance of a sentence with SRL data.

				:param tokens: a list of strings
				:param arg_structures: a list of tuples in the format (predicate, mapping).
						Each predicate is a string and each mapping is a dictionary mapping role labels
						to the words that constitute it.
				"""
				self.tokens = tokens
				self.arg_structures = arg_structures

class ParsedSentence(object):
		"""
		Class for storing a sentence with dependency parsing annotation.

		It stores a list of tokens, the dependency heads, dependency labels and POS tags
		if the parser used them. Dependency heads are the index of the head of each
		token, and -1 means a dependency to the root.
		"""
		def __init__(self, tokens, heads, labels, pos=None):
				"""
				Constructor.

				:param tokens: list of strings
				:param heads: list of integers (-1 means dependency to root, others are token indices)
				:param labels: list of strings
				:param pos: None or list of strings
				"""
				self.tokens = tokens
				self.heads = heads
				self.labels = labels
				self.pos = pos

		def __len__(self):
				return len(self.tokens)

		def to_conll_list(self):
				"""
				Return a list representation of the sentence in CoNLL X format.

				Each line has:
				[number starting from 1] token _ POS POS _ head label

				Token numbers start from 1, root is referred as 0.
				POS is only available if the original parser used it.
				"""
				tokenL = []
				headL = []
				labelL = []
				posL = []
				for i in range(len(self.tokens)):
						tokenL.append(self.tokens[i])
						headL.append(self.heads[i] + 1)
						labelL.append(self.labels[i])
						posL.append(self.pos[i])

				return tokenL, posL, labelL, headL

		def to_conll(self):
				"""
				Return a string representation of the sentence in CoNLL X format.

				Each line has:
				[number starting from 1] token _ POS POS _ head label

				Token numbers start from 1, root is referred as 0.
				POS is only available if the original parser used it.
				"""
				result = []
				for i in range(len(self.tokens)):
						token = self.tokens[i]
						head = self.heads[i] + 1
						label = self.labels[i]
						pos = self.pos[i] if self.pos else '_'

						#line = u'{id}\t{token}\t_\t{pos}\t{pos}\t_\t{head}\t{label}'
						#result.append(line.format(id=i+1, pos=pos, head=head, label=label, token=token))
						line = u'{id}\t{token}\t{head}\t{label}'
						result.append(line.format(id=i+1, head=head, label=label, token=token))

				return '\n'.join(result)


class Tagger(object):
		"""
		Base class for taggers. It should not be instantiated.
		"""
		def __init__(self, data_dir=None):
				"""Creates a tagger and loads data preemptively"""
				asrt_msg = "espresso data directory is not set. \
If you don't have the trained models, download them from http://air.cwnu.ac.kr/espresso/models.html"
				if data_dir is None:
						assert config.data_dir is not None, asrt_msg
						self.paths = config.FILES
				else:
						self.paths = config.get_config_paths(data_dir)

				self.data_dir = data_dir
				self._load_data()

		def _load_data(self):
				"""Implemented by subclasses"""
				pass


class SRLTagger(Tagger):
		"""
		An SRLTagger loads the models and performs SRL on text.

		It works on three stages: predicate identification, argument detection and
		argument classification.
		"""

		def _load_data(self):
				"""Loads data for SRL"""
				md_srl = Metadata.load_from_file('srl', self.paths)
				self.nn = load_network(md_srl)
				self.reader = create_reader(md_srl)
				self.reader.create_converter()
				self.itd = self.reader.get_inverse_tag_dictionary()

				self.parser = DependencyParser(self.data_dir)


		def find_predicates(self, tokens):
				"""
				Finds out which tokens are predicates.

				:param tokens: a list of attribute.Token elements
				:returns: the indices of predicate tokens
				"""
				answer = []
				for i, token in enumerate(tokens):
					if token[0] == 'V' and tokens[i-1][0] != 'V': answer.append(i+1)
				return np.array(answer)

		def find_arguments(self, token_obj, predL, headL, relL):
				"""
				Finds out which tokens are predicates.

				:param tokens: a list of attribute.Token elements
				:returns: the indices of predicate tokens
				"""
				answer_token = []; answer = []
				for p in predL:
					pred_arg_token = []; pred_arg = []
					for j, h in enumerate(headL):
						if p == h and relL[j][0] == 'N':
							pred_arg_token.append(token_obj[j])
							pred_arg.append(np.array([j, j]))

					#TODO
					# predicate의 header
					#if headL[p-1] != 0: # 마지막 제외
					#	pred_arg_token.append(token_obj[headL[p-1]-1])
					#	pred_arg.append(np.array([headL[p-1]-1, headL[p-1]]))

					answer_token.append(pred_arg_token)
					answer.append(pred_arg)
				#print(answer_token)
				#print(answer)
				return answer_token, answer

		def tag(self, text, use_sent_tokenizer=True, mode='standard'):
				"""
				Runs the SRL process on the given text.

				:param text: unicode or str encoded in utf-8.
				:param no_repeats: whether to prevent repeated argument labels
				:returns: a list of SRLAnnotatedSentence objects
				"""
				tokens = utils.tokenize(text)
				result = []
				for sent in tokens:
						tagged = self.tag_sentence(sent)
						result.append(tagged)

				return result

		def tag_sentence(self, tokens, no_repeats=False):
				"""
				Runs the SRL process on the given tokens.

				:param tokens: a list of tokens (as strings)
				:param no_repeats: whether to prevent repeated argument labels
				:returns: a list of lists (one list for each sentence). Sentences have tuples
						(all_tokens, predicate, arg_structure), where arg_structure is a dictionary
						mapping argument labels to the words it includes.
				"""
				# 구문분석
				parsed = self.parser.parse_sentence(tokens)
				wordL, posL, relL, headL =  parsed.to_conll_list()
				tokens_obj = []
				for w, p, r in zip(wordL, posL, relL):
					hm, hp, tm, tp = p
					token = attributes.Token(w, hm, hp, tm, tp, r)
					tokens_obj.append(token)

				converted_class = np.array([self.reader.converter.convert(t)
																		for t in tokens_obj])
				pred_positions = self.find_predicates(relL)

				arg_tokens, arg_limits = self.find_arguments(wordL, pred_positions, headL, relL)
				print(arg_tokens)
				print(pred_positions)
				print(arg_limits)

				# now, argument classification
				answers = self.nn.tag_sentence(converted_class,
																								pred_positions, arg_limits,
																								allow_repeats=not no_repeats)
				labels = [[self.itd[x] for x in pred_answer]
										 for pred_answer in answers]

				structures = _group_arguments(wordL, pred_positions, arg_tokens, labels)
				return SRLAnnotatedSentence(wordL, structures)

class DependencyParser(Tagger):
		"""A Dependency Parser based on a neural network tagger."""

		def __init__(self, *args, **kwargs):
				"""
				Set the data directory for the POS tagger, if one is used,
				and call the parent constructor.
				"""
				super(DependencyParser, self).__init__(*args, **kwargs)

		def _load_data(self):
				"""Loads data for Dependency Parsing"""
				md_udep = Metadata.load_from_file('unlabeled_dependency', paths=self.paths)
				self.unlabeled_nn = load_network(md_udep)
				self.unlabeled_reader = create_reader(md_udep)

				md_ldep = Metadata.load_from_file('labeled_dependency', paths=self.paths)
				self.labeled_nn = load_network(md_ldep)
				self.labeled_reader = create_reader(md_ldep)
				self.itd = self.labeled_reader.get_inverse_tag_dictionary()

				self.use_pos = md_udep.use_pos or md_ldep.use_pos
				if self.use_pos:
						self.pos_tagger = POSTagger(self.data_dir)

		def parse(self, text):
				"""
				Split the given text into sentences and determines their
				dependency trees. If you want to provide your own tokenized
				text, use `parse_sentence` instead.

				:param text: a string
				:returns: a list of ParsedSentence's
				"""
				sentences = utils.tokenize(text)
				result = []
				for sent in sentences:
						parsed = self.parse_sentence(sent)
						result.append(parsed)

				return result

		def tag_tokens(self, tokens):
			"""
			Parse the given sentence. This function is just an alias for
			`parse_sentence`.
			"""
			return self.parse_sentence(tokens)

		def parse_sentence(self, tokens):
			"""
			Parse the given sentence. It must be already tokenized; if you
			want nlpnet to tokenize the text, use the method `parse` instead.

			:param tokens: a list of strings (sentences)
			:return: a ParsedSentence instance
			"""
			original_tokens = tokens
			udep_tokens_obj = []
			ldep_tokens_obj = []

			# if the parser uses POS a feature, have a tagger tag it first
			if self.use_pos:
				eojeols, eojeol_features = self.pos_tagger.tag_tokens(tokens, mode='eojeol')
				#print("**", eojeols)
				#print(eojeol_features)
			#print(tokens, eojeols)

			for word, feature in zip(eojeols, eojeol_features):
				m_h, t_h, m_t, t_t = feature
				#udep_tokens_obj.append(attributes.Token(word, morph_h=m_h, pos_h=t_h, morph_t=m_t, pos_t=t_t))
				udep_tokens_obj.append(attributes.Token(word, pos_h=t_h, morph_t=m_t, pos_t=t_t))
				ldep_tokens_obj.append(attributes.Token(word, pos_h=t_h, morph_t=m_t, pos_t=t_t))

			converted_tokens = self.unlabeled_reader.codify_sentence(udep_tokens_obj)
			#print(converted_tokens)
			heads = self.unlabeled_nn.tag_sentence(converted_tokens)
			#print(heads)

			# the root is returned having a value == len(sentence)
			root = heads.argmax()
			heads[root] = root

			converted_tokens = self.labeled_reader.codify_sentence(ldep_tokens_obj)
			label_codes = self.labeled_nn.tag_sentence(converted_tokens, heads)
			labels = [self.itd[code] for code in label_codes]
			#print(label_codes)
			#print(labels)

			# to the final answer, signal the root with -1
			heads[root] = -1
			pos_tags = eojeol_features if self.use_pos else None
			#pos_tags = zip(*tokens)[1] if self.use_pos else None

			parsed = ParsedSentence(eojeols, heads, labels, pos_tags)
			#parsed = ParsedSentence(original_tokens, heads, labels, pos_tags)
			return parsed

		def tag(self, text, use_sent_tokenizer=True, mode='eojeol'):
			"""
			Parse the given text. This is just an alias for the
			`parse` method.
			"""
			return self.parse(text)


class WSDTagger(Tagger):
		"""A WSDTagger loads the models and performs WSD tagging on text."""

		def _load_data(self):
				"""Loads data for WSD"""
				md_wsd = Metadata.load_from_file('wsd', self.paths)
				self.nn = load_network(md_wsd)
				self.reader = create_reader(md_wsd)
				self.reader.create_converter()
				self.itd = self.reader.get_inverse_tag_dictionary()
				#self.morph_lexicon = self.reader.morph_lexicon					# user lexicon
				#self.co_lexicon = self.reader.co_lexicon
				#self.prob_dict = self.reader.prob_dict
				self.pos_tagger = POSTagger(self.data_dir)

		def tag(self, text, use_sent_tokenizer=True, mode='standard'):
				"""
				Tags the given text.

				:param text: a string or unicode object. Strings assumed to be utf-8
				:returns: a list of lists (sentences with tokens).
						Each sentence has (token, tag) tuples.
				"""
				tokens = utils.tokenize(text)
				result = []
				for sent in tokens:
					tagged = self.tag_sentence(sent)
					result.append(tagged)

				return result

		def tag_sentence(self, tokens):
			"""
			Tags a given list of tokens.

			Tokens should be produced with the espresso tokenizer in order to
			match the entries in the vocabulary. If you have non-tokenized text,
			use NERTagger.tag(text).

			:param tokens: a list of strings
			:returns: a list of strings (morphs, tags)
			"""
			pos_tagged = self.pos_tagger.tag_tokens(tokens)

			pos_tagged = filter(lambda x : x != (' ', 'SP'), pos_tagged)
			unzipped_pos_tagged = zip(*pos_tagged)
			morphs, morph_pos_tags = list(unzipped_pos_tagged)
			#print(morphs, morph_pos_tags)

			converter = self.reader.converter
			converted_tokens = np.array([converter.convert(token) for token in morphs])
			#print("0", converted_tokens)

			answer = self.nn.tag_sentence(converted_tokens)
			tags = [self.itd[tag] for tag in answer]				# 번호를 수로 표현

			#print("1", morphs, tags)

			return zip(morphs, tags)


class NERTagger(Tagger):
		"""A NERTagger loads the models and performs NER tagging on text."""

		def _load_data(self):
				"""Loads data for NER"""
				md_ner = Metadata.load_from_file('ner', self.paths)
				self.nn = load_network(md_ner)
				self.reader = create_reader(md_ner)
				self.reader.create_converter()
				self.itd = self.reader.get_inverse_tag_dictionary()
				#self.morph_lexicon = self.reader.morph_lexicon					# user lexicon
				#self.co_lexicon = self.reader.co_lexicon
				#self.prob_dict = self.reader.prob_dict
				self.pos_tagger = POSTagger(self.data_dir)

		def tag(self, text, use_sent_tokenizer=True, mode='standard'):
				"""
				Tags the given text.

				:param text: a string or unicode object. Strings assumed to be utf-8
				:returns: a list of lists (sentences with tokens).
						Each sentence has (token, tag) tuples.
				"""
				tokens = utils.tokenize(text)
				result = []
				for sent in tokens:
					tagged = self.tag_sentence(sent)
					result.append(tagged)

				return result

		def tag_sentence(self, tokens):
			"""
			Tags a given list of tokens.

			Tokens should be produced with the espresso tokenizer in order to
			match the entries in the vocabulary. If you have non-tokenized text,
			use NERTagger.tag(text).

			:param tokens: a list of strings
			:returns: a list of strings (morphs, tags)
			"""
			pos_tagged = self.pos_tagger.tag_tokens(tokens)

			pos_tagged = filter(lambda x : x != (' ', 'SP'), pos_tagged) # 공백 제거
			unzipped_pos_tagged = zip(*pos_tagged)
			morphs, morph_pos_tags = list(unzipped_pos_tagged)
			#print(morphs, morph_pos_tags)

			converter = self.reader.converter
			converted_tokens = np.array([converter.convert(token) for token in morphs])
			#print("0", converted_tokens)

			answer = self.nn.tag_sentence(converted_tokens)
			tags = [self.itd[tag] for tag in answer]				# 번호를 수로 표현

			#print("1", morphs, tags)

			return zip(morphs, tags)


class POSTagger(Tagger):
		"""A POSTagger loads the models and performs POS tagging on text."""

		def _load_data(self):
				"""Loads data for POS"""
				md = Metadata.load_from_file('pos', self.paths)
				self.nn = load_network(md)
				self.reader = create_reader(md)
				self.reader.create_converter()
				self.itd = self.reader.get_inverse_tag_dictionary()
				self.morph_lexicon = self.reader.morph_lexicon					# user lexicon
				self.co_lexicon = self.reader.co_lexicon
				self.prob_dict = self.reader.prob_dict

		def tag(self, text, use_sent_tokenizer=True, mode="standard"):
				"""
				Tags the given text.

				:param text: a string or unicode object. Strings assumed to be utf-8
				:param mode:  [standard, eumjeol, eojeol]. "eumjeol" does not lemmatize,
						"eojeol" includes NN+XV
				:returns: a list of lists (sentences with tokens).
						Each sentence has (token, tag) tuples.
				"""
				tokens = utils.tokenize(text, use_sent_tokenizer) # 문장 단위 + 음절
				result = []
				for sent in tokens:
						tagged = self.tag_tokens(sent, mode)
						result.append(tagged)

				return result

		def tag_tokens(self, tokens, mode="standard"):
				"""
				Tags a given list of tokens.

				Tokens should be produced with the espresso tokenizer in order to
				match the entries in the vocabulary. If you have non-tokenized text,
				use POSTagger.tag(text).

				:param tokens: a list of strings
				:param mode: [standard, eumjeol, verb]. "eumjeol" does not lemmatize,
						"verb" includes NN+XV
				:returns: a list of strings (the tags)
				"""
				converter = self.reader.converter			# 클래스 지정
				converted_tokens = np.array([converter.convert('*space*') if token==' ' else converter.convert(token)
																		 for token in tokens])
				#print("0", converted_tokens)

				answer = self.nn.tag_sentence(converted_tokens)
				tags = [self.itd[tag] for tag in answer]				# 번호를 문자로 변환

				if mode == 'eojeol':
					eojeols, eojeol_features = self.get_eojeol_tokens(tokens, tags, mode)
					return eojeols, eojeol_features
				else:
					morphs, morph_tags = self.get_morph_tokens(tokens, tags, mode)
					return zip(morphs, morph_tags)


		def _get_morph_tokens(self, tokens, tags):
			"""
			공백으로 형태소 분리.

			:param tokens: a list of strings
			:param tags: a list of tags of each string
			:return: a list of (morph, tag)
			"""
			#print(utils.get_word(self.morph_lexicon, tokens, tags, True))
			# 기분석 사전 검색
			tokens, tags = utils.get_word(self.morph_lexicon, tokens, tags, True)
			#print(tokens)
			#print(tags)
			morphs = [''.join(tokens[0]) if isinstance(tokens[0], list) else tokens[0]]
			morph_tags = [(lambda x: 'MA' if x == 'MS' else x)\
						((lambda x: 'NN' if x == 'NS' else x)(tags[0]))]
			for idx in range(1,len(tokens)):
				if (tags[idx-1]=='NS' and tags[idx]=='NN') \
						or (tags[idx-1]=='MS' and tags[idx]=='MA'):
					morphs.append(morphs.pop()+(''.join(tokens[idx]) if isinstance(tokens[idx], list) else tokens[idx]))
				elif tags[idx-1] != tags[idx] or tags[idx] == 'SY':
					morphs.append(''.join(tokens[idx]) if isinstance(tokens[idx], list) else tokens[idx])
					morph_tags.append((lambda x: 'MA' if x == 'MS' else x)\
						((lambda x: 'NN' if x == 'NS' else x)(tags[idx])))
				else:
					morphs.append(morphs.pop()+(''.join(tokens[idx]) if isinstance(tokens[idx], list) else tokens[idx]))

			return morphs, morph_tags


		def get_eumjeol_tokens(self, tokens, tags):
			"""
			음절 토큰으로 처리.
			'CO'를 앞 형태소에 붙이고 품사는 앞의 것을 따름
			새로운 -> 새/VB+로운/CO -> 새로운/VB

			:param tokens: a list of strings
			:param tags: a list of tags of each string
			:return: a list of (eumjeol, tag)
			"""
			eumjeol = []
			eumjeol_tags = []
			#print(tokens)
			#print(tags)
			for idx in range(0, len(tokens)):
				if idx>0 and (tags[idx]=='CO' and \
						tags[idx-1]!='SP' and tags[idx-1][1]!='N'):
					eumjeol.append(eumjeol.pop()+(''.join(tokens[idx]) if isinstance(tokens[idx], list) else tokens[idx]))
				elif idx>0 and (tags[idx]=='CO' and \
						tags[idx-1]!='SP' and tags[idx-1][1]=='N'):
					eumjeol.append(tokens[idx])
					eumjeol_tags.append('XV')
				elif tags[idx] =='CO':
					eumjeol.append(tokens[idx])
					eumjeol_tags.append('VB')
				else:
					eumjeol.append(tokens[idx])
					eumjeol_tags.append(tags[idx])
			#print(eumjeol)
			#print(eumjeol_tags)

			return eumjeol, eumjeol_tags

		def get_eojeol(self, tokens, tags):
			"""
			음절 토큰으로 처리.
			'CO'를 앞 형태소에 붙이고 품사는 앞의 것을 따름
			새로운 -> 새/VB+로운/CO -> 새로운/VB

			:param tokens: a list of strings
			:param tags: a list of tags of each string
			:return: a list of (eumjeol, tag)
			"""
			eojeols = []
			eumjeol = []
			#print(tokens)
			for t in tokens:
				if t == ' ':
					eojeols.append(''.join(eumjeol))
					eumjeol = []
				else:
					eumjeol.append(t)
			eojeols.append(''.join(eumjeol))
			#print(eojeols)

			return eojeols



		def get_morph_tokens(self, tokens, tags, mode="standard"):
			"""
			combine eumjeol to morph

			param tokens: eumjeol token list
			param tags: pos tag list of each token
			"""
			_morphs, _morph_tags = self._get_morph_tokens(tokens, tags)
			#print('2---', morphs, morph_tags) # 원형 복원 전

			if mode=='eumjeol':
				eumjeols, eumjeol_tags = self.get_eumjeol_tokens(_morphs, _morph_tags)
				return eumjeols, eumjeol_tags

			# 'CO', 축약어 처리
			morphs, morph_tags = self.handling_abbrs(_morphs, _morph_tags)
			#print("3", morphs, morph_tags)  # 원형복원

			if mode=='eojeol':
				eojeols = self.get_eojeol(_morphs, _morph_tags)
				return eojeols, morphs, morph_tags
			return morphs, morph_tags

		def handling_abbrs(self, _morphs, _tags):
			'''
			CO tag와 축약을 처리한다.
			'''
			morphs = []
			morph_tags = []
			#print(_morphs, _tags, flush=True)

			for i, t in enumerate(_tags):
				if t == 'CO':
					prev_morph = _morphs[i-1] if i > 0 else 'BOS'
					prev_tag = _tags[i-1] if i > 0 else 'BOS'
					next_morph = _morphs[i+1] if i < len(_tags)-1 else 'EOS'
					next_tag   = _tags[i+1] if i < len(_tags)-1 else 'EOS'
					if _tags[i-1] in ['MM']:
						morph_tags[-1] = 'NN'; _tags[i-1] = 'NN'; prev_tag='NN'  # 나중 제거할 것
					morphs, morph_tags =  self.handling_co_tags(morphs, morph_tags, \
						prev_morph, prev_tag, _morphs[i], _tags[i], next_morph, next_tag)
				elif i > 0:
					morphs, morph_tags = self.handling_others(morphs, morph_tags, _morphs[i], _tags[i])
				else:
					if _morphs[i] == ' ': t = 'SP'
					morphs.append(_morphs[i])
					morph_tags.append(t)
			return morphs, morph_tags


		def handling_others(self, morphs, morph_tags, morph, tag):
			'''
			다른 형태소들의 처리를 담당한다. 여기에는
			1. 'ㄴ,ㄹ, ㅁ, ㅂ, ㅆ' 의 축약
			2. '가수다'와 같은 어절이 포함된다.
			'''
			#print(morphs, morph_tags, morph, tag)
			try:
				if morph_tags[-1] == tag:  # 이미 복원된 것과 연결
					morphs.append(morphs.pop()+morph) # 미룬다 -> (미루 + ㄴ) + 다 -> 미루 + ㄴ다
				elif morph_tags[-1] == 'NN' and tag == 'EE': # '가수다'
					morphs.append('이')
					morph_tags.append('VB')
					morphs.append(morph)
					morph_tags.append(tag)
				elif morph_tags[-1] == 'MM' and tag == 'XV': # 제거 할 것, 'MMXV' 대응
					morph_tags[-1] = 'NN'
					morphs.append(morph)
					morph_tags.append(tag)
				else:
					morphs.append(morph)
					morph_tags.append(tag)
					#print('9>', i, morphs, morph_tags)
			except:
				print('>>>', morphs, morph, tag)
			return morphs, morph_tags

		def handling_co_tags(self, morphs, morph_tags, m_1, t_1, m, t, m__1, t__1):
			"""
			CO tag를 다룬다. CO tag의 형태소를 확장한다.
			"""
			#print(morphs, morph_tags, m, t)
			#------------------------------------------------------------
			def get_best_path(l):
				max_p = -1000; max_list = []; max_same_morph = 10
				for idx, x in enumerate(l):
					same_morph = 0
					_m_t_ = etc.parse_morph(x) # '가/VB+ㄴ/EE' -> [(가, VB), (ㄴ, EE)]
					#print(morphs, morph_tags, m_1, t_1, _m_t_)
					# 이전 형태소
					if (t_1 == _m_t_[0][1]) or (t_1 in ['JJ']):
						same_morph = -1 if len(morph_tags)>1 else 0
						first_word = _m_t_[0][0]
						#print(same_morph, morph_tags[same_morph])
						while morph_tags[same_morph] == _m_t_[0][1] or morph_tags[same_morph] in ['JJ']:
							first_word = morphs[same_morph] + first_word
							same_morph -= 1
							if (len(morph_tags)+same_morph)<0: break
						prev_word = (morphs[same_morph]+'/'+morph_tags[same_morph]) if (len(morph_tags)+same_morph)>=0 else 'BOS'
						prev_tag = morph_tags[same_morph] if (len(morph_tags)+same_morph)>=0 else 'BOS'
						first_word = first_word+'/'+_m_t_[0][1]
					else:
						prev_word = m_1+'/'+t_1 # viterbi를 위해서
						prev_tag = t_1
						first_word = (_m_t_[0][0]+'/'+_m_t_[0][1])

					first_tag = _m_t_[0][1]
					last_word = _m_t_[-1][0]+'/'+_m_t_[-1][1]
					last_tag = _m_t_[-1][1]
					p = (self.prob_dict[prev_word] if prev_word in self.prob_dict else -100)
					#print(p)
					p += (self.prob_dict[prev_tag + '' + first_tag] if prev_tag + '' + first_tag in self.prob_dict else -100) \
						+ (self.prob_dict[first_word] if first_word in self.prob_dict else -100) \
						+ (self.prob_dict[last_word] if last_word in self.prob_dict else -100) \
						+ (self.prob_dict[last_tag + '' + t__1] if last_tag + '' + t__1 in self.prob_dict else -100)
					#print(p)
					if p > max_p:
						max_p = p
						max_same_morph = same_morph
						max_list = []
						for i, (m, t) in enumerate(_m_t_):
							m = first_word.split('/')[0] if i==0 else m
							t = first_tag if i==0 else t
							max_list.append((m,t))
						#max_list = _m_t_
				#print(max_same_morph, max_list)
				return max_list, max_same_morph+1

			# ---------------------------------------------------------
			try:
				l = self.co_lexicon[m].split('|')
			except:
				morphs.append(m)
				morph_tags.append('NN')
				return morphs, morph_tags

			if len(l) == 1: # 후보가 하나일 경우
				_m_t_ = etc.parse_morph(l[0])
				for _m_, _t_ in _m_t_:
					if len(morph_tags)>1 and morph_tags[-1] == _t_:
						morphs.append(morphs.pop()+_m_)
					else:
						morphs.append(_m_)
						morph_tags.append(_t_)
				return morphs, morph_tags

			# ------- 사전에 후보가 2개 이상일 경우 ----------
			max_list, overlap_idx = get_best_path(l)

			co_morphs = [m for (m,t) in max_list]
			co_morph_tags = [t for (m,t) in max_list]
			#print(':::', overlap_idx, morphs, co_morphs)
			if overlap_idx < 0:
				morphs = morphs[:overlap_idx] + co_morphs
				morph_tags = morph_tags[:overlap_idx] + co_morph_tags
			else:
				morphs = morphs + co_morphs
				morph_tags = morph_tags + co_morph_tags

			return morphs, morph_tags

		def get_eojeol_tokens(self, tokens, tags, mode="eojeol"):
			"""
			# 복원 후 떨어진 형태소 연결, 구문분석에서 XV 형태소 연결하기
			# 사랑+하 -> 사랑하 (구문분석)

			param tokens : 음절
			param tags   : 품사
			"""
			eojeols, morphs, morph_tags = self.get_morph_tokens(tokens, tags, mode)
			eojeol_features = []
			#print(morphs, morph_tags)
			head_m = ''; head_t = ''; tail_m=''; tail_t=''
			for i in range(len(morphs)):
				t = morph_tags[i]
				#print(i, morphs[i], len(morphs), t)
				#if ((i==0 or morphs[i-1] == ' ') and (i == len(morphs)-1 or morphs[i+1] == ' ')):
				#	# 기호로만 한 어절이 되는 경우
				#	head_m = morphs[i]
				#	head_t = morph_tags[i]
				#	tail_m = morphs[i]
				#	tail_t = morph_tags[i]
				#	eojeol_features.append((head_m, head_t, tail_m, tail_t))
				#	print("1::::", head_m, head_t, tail_m, tail_t)
				#	continue

				# 어절 마지막
				if t == 'SP' :
					## tail feature of last eojeol
					tail_m = morphs[i-2] if (morph_tags[i-1] == 'SY' and morphs[i-1]!=',' and morphs[i-2]!= ' ') else morphs[i-1]
					tail_t = morph_tags[i-2] if (morph_tags[i-1] == 'SY' and morphs[i-1]!=',' and morphs[i-2] != ' ') else morph_tags[i-1]
					tail_t = 'EE' if tail_t in ['XV', 'VB'] else tail_t		# 비상 종결인 경우
					eojeol_features.append((head_m, head_t, tail_m, tail_t))
					#print("2::::", head_m, head_t, tail_m, tail_t)
					continue

				if i == len(morphs)-1:
					## tail feature of last eojeol
					tail_m = morphs[i-1] if (morph_tags[i] == 'SY' and morphs[i]!=',' and morphs[i-1]!= ' ') else morphs[i]
					tail_t = morph_tags[i-1] if (morph_tags[i] == 'SY' and morphs[i]!=',' and morphs[i-1]!= ' ') else morph_tags[i]
					tail_t = 'EE' if tail_t in ['XV', 'VB'] else tail_t		# 비상 종결인 경우
					eojeol_features.append((head_m, head_t, tail_m, tail_t))
					#print("3::::", head_m, head_t, tail_m, tail_t)
					continue

				# 어절 처음
				if i == 0 or morphs[i-1] == ' ':
					head_m = morphs[i+1] if (morph_tags[i] == 'SY' and morph_tags[i+1] != 'SP') else morphs[i]
					head_t = morph_tags[i+1] if (morph_tags[i] == 'SY' and morph_tags[i+1] != 'SP') else morph_tags[i]
					idx = 2 if ((morph_tags[i] == 'SY' and morph_tags[i+1] != 'SP') and i < len(morphs)-2) else 1
					head_t += morph_tags[i+idx] if morph_tags[i+idx] in ['XV', 'VB'] else ''
					#print("4:::", i, idx, morph_tags, head_m, head_t, tail_m, tail_t)

			#print(eojeols, eojeol_features)
			return eojeols, eojeol_features
