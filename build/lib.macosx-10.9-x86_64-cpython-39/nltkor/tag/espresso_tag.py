import os

# import sys
# sys.path.append("/Users/dowon/nltk_ko/nltk/tag")
# from libs import *

"""
This script will run a POS or SRL tagger on the input data and print the results
to stdout.
"""

import argparse
import logging

if __package__ is None: # 스크립트로 실행할 때
		import sys
		from os import path
		#print(path.dirname( path.dirname( path.abspath(__file__) ) ))
		sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
		from libs import *
else:
	from .libs import *
#from .libs import *
import requests
import zipfile

logging.disable(logging.INFO) # disable INFO and DEBUG logging everywhere


class EspressoTagger:
		def __init__(self, data_dir=None, task="pos"):
				self.data_dir = data_dir
				if data_dir == None:
						path=os.path.dirname(__file__)
						path= path + '/data'
						self.data_dir = path
						#print(path)
				self.path = ""
				self.tagger = None

				self.task = task.lower()
				if not self._check_model():
						self._download_model()

				set_data_dir(self.data_dir)

				if self.task == 'pos':
								self.tagger = taggers.POSTagger(data_dir=self.data_dir)
				elif self.task == 'ner':
								self.tagger = taggers.NERTagger(data_dir=self.data_dir)
				elif self.task == 'wsd':
								self.tagger = taggers.WSDTagger(data_dir=self.data_dir)
				elif self.task == 'srl':
								self.tagger = taggers.SRLTagger(data_dir=self.data_dir)
				elif self.task == 'dependency':
								self.tagger = taggers.DependencyParser(data_dir=self.data_dir)
				else:
								raise ValueError('Unknown task: %s' % self.task)


		def tag(self, text, use_sent_tokenizer=True, lemma=True):
				"""
				This function provides an interactive environment for running the system.
				It receives text from the standard input, tokenizes it, and calls the function
				given as a parameter to produce an answer.

				:param task: 'pos', ner', 'wsd', 'srl' or 'dependency'
				:param use_tokenizer: whether to use built-in tokenizer
				"""

				#use_sent_tokenizer = not use_sent_tokenizer
				mode = 'standard' if lemma else 'eumjeol'
				result = self.tagger.tag(text, use_sent_tokenizer, mode)
				'''
				else:
								tokens = text.split()
								if self.task != 'dependency':
												result = [self.tagger.tag_tokens(tokens, True)]
								else:
												result = [self.tagger.tag_tokens(tokens)]
				'''

				return self._result_tagged(result, self.task)

		def _result_tagged(self, tagged_sents, task):
				"""
				Prints the tagged text to stdout.

				:param tagged_sents: sentences tagged according to any of espresso taggers.
				:param task: the tagging task (either 'pos', 'ner', 'wsd', 'srl' or 'dependency')
				"""

				##TODO: print부분 return으로 변경
				if task == 'pos':
								return self._return_tagged_pos(tagged_sents)
				elif task == 'ner':
								return self._return_tagged_ner(tagged_sents)
				elif task == 'wsd':
								return self._return_tagged_wsd(tagged_sents)
				elif task == 'srl':
								return self._return_tagged_srl(tagged_sents)
				elif task == 'dependency':
								return self._return_parsed_dependency(tagged_sents)
				else:
								raise ValueError('Unknown task: %s' % task)



		def _return_parsed_dependency(self, parsed_sents):
				"""Prints one token per line and its head"""
				result = []
				temp_list = []
				temp_list2 = []
				for sent in parsed_sents:
						temp_list = sent.to_conll().split('\t')
						temp_list = temp_list[1:]
						for ele in temp_list:
								if '\n' in ele:
										ele = ele[:ele.find('\n')]
								temp_list2.append(ele)
						result.append(self._dependency_after(temp_list2)[:])
						temp_list2 = []

				return result

		def _return_tagged_pos(self, tagged_sents):
			"""Prints one sentence per line as token_tag"""
			result = []
			for sent in tagged_sents:
				result = result + list(sent)
			return result

		def _return_tagged_srl(self, tagged_sents):
			result = []
			for sent in tagged_sents:
				# print (' '.join(sent.tokens))
				temp_dict1 = {}
				for predicate, arg_structure in sent.arg_structures:
					# print ("test 1 :", predicate)
					# print("te22 :", arg_structure)

					temp_dict2 = {}
					for label in arg_structure:
						argument = ' '.join(arg_structure[label])
						# line = '\t%s: %s' % (label, argument)
						# print (line)
						temp_dict2[label] = argument

						# result.append((label, argument))
						# print ('\n')
					temp_dict1[predicate] = temp_dict2

				result.append(temp_dict1)

			return result

		def _return_tagged_ner(self, tagged_sents):
				"""Prints one sentence per line as token_tag"""
				result = []
				for sent in tagged_sents:
						for item in sent:
								#s = '_'.join(item)
								result.append(item)

				return result

		def _return_tagged_wsd(self, tagged_sents):
				"""Prints one sentence per line as token_tag"""
				result = []
				for sent in tagged_sents:
						for item in sent:
								s = '_'.join(item)
								result.append(s)

				return result

		def _download_model(self):
				"""Downloads the model from the server"""
				temp_path = os.path.dirname(__file__) + '/data.zip'
				url = "https://air.changwon.ac.kr/~airdemo/storage/espresso_data_1/data.zip"
				print("Downloading Espresso5 model...")
				with requests.get(url, stream=True) as r:
						r.raise_for_status()
						with open(temp_path, 'wb') as f:
								for chunk in r.iter_content(chunk_size=8192):
								# If you have chunk encoded response uncomment if
								# and set chunk_size parameter to None.
								#if chunk:
										f.write(chunk)

				if os.path.exists(self.data_dir):
						os.rmdir(self.data_dir)

				with zipfile.ZipFile(temp_path, "r") as zip_ref:
						zip_ref.extractall(os.path.dirname(__file__))

		def _check_model(self):
				"""Checks if the model is available and downloads it if necessary"""
				if not os.path.exists(self.data_dir):
						return False
				else:
						return True

		def _dependency_after(self, list):
				len_list = len(list)
				temp_list = []
				repeat = len_list//3
				for i in range(repeat):
						index = i*3
						tup1 = (i+1, )
						tup2 = tuple(list[index:index+3])
						tup = tup1 + tup2
						temp_list.append(tup[:])


				return temp_list

if __name__ == '__main__':
		tagger = EspressoTagger(task='pos')
		print(tagger.tag("나는 아름다운 강산에 살고있다."))
