from collections import defaultdict
from nltk.translate.bleu_score import *
from nltk.metrics import scores
from nltk.metrics import confusionmatrix
from nltkor.tokenize import word_tokenize,sent_tokenize,syllable_tokenize
from nltk.util import ngrams, skipgrams
from nltkor.cider.cider import Cider
import sys
import itertools
from nltkor.tag import pos_tag, pos_tag_with_verb_form, EspressoTagger
from nltkor.sejong import ssem
from typing import Callable, Iterable, List, Tuple
from copy import deepcopy

# Natural Language Toolkit: Machine Translation
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Uday Krishna <udaykrishna5@gmail.com>
# Contributor: Tom Aarsen
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

class StringMetric:
	def __init__(self, lang="korean"):
		self.lang = lang
		if lang not in ["korean", "english"]:
			raise Exception("Only \"korean\" or \"english\" in lang parameter")
		self.tokenize=lambda ref: word_tokenize(ref,self.lang)
		self.skip_pos = ['EE']


	def _W_CER(self, r, h):

		costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

		DEL_PENALTY=1 # Tact
		INS_PENALTY=1 # Tact
		SUB_PENALTY=1 # Tact

		for i in range(1, len(r)+1):
			costs[i][0] = DEL_PENALTY*i

		for j in range(1, len(h) + 1):
			costs[0][j] = INS_PENALTY*j

		# computation
		for i in range(1, len(r)+1):
			for j in range(1, len(h)+1):
				if r[i-1] == h[j-1]:
					costs[i][j] = costs[i-1][j-1]
				else:
					substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
					insertionCost	= costs[i][j-1] + INS_PENALTY   # penalty is always 1
					deletionCost	 = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

					costs[i][j] = min(substitutionCost, insertionCost, deletionCost)

		mo = len(r)
		i = len(r)
		j = len(h)

		result=(costs[i][j])/mo

		if result>1.0:
			return 1.0
		else:
			return result


	def wer(self, reference, candidate):
		r = word_tokenize(reference, self.lang)
		h = word_tokenize(candidate, self.lang)

		return self._W_CER(r,h)


	def cer(self, reference,candidate):
		r = syllable_tokenize(reference, self.lang)
		h = syllable_tokenize(candidate, self.lang)

		return self._W_CER(r,h)


	def bleu(self, reference, candidate,weights=(0.25,0.25,0.25,0.25), smoothing_function=None):

		if type(candidate)!=list or type(reference)!=list:
			print("parameter expect list type")
			return

		reference=list(map(self.tokenize,reference))
		candidate=word_tokenize(candidate,self.lang)

		return sentence_bleu(reference,candidate,weights,smoothing_function=smoothing_function)


	def bleu_n(self, reference,candiate,n=1, smoothing_function=None):

		if n==1:
			return self.bleu(reference,candiate,(1,0,0,0), smoothing_function=smoothing_function)
		elif n==2:
			return self.bleu(reference,candiate,(0,1,0,0), smoothing_function=smoothing_function)
		elif n==3:
			return self.bleu(reference,candiate,(0,0,1,0), smoothing_function=smoothing_function)
		elif n==4:
			return self.bleu(reference,candiate,(0,0,0,1), smoothing_function=smoothing_function)




	def _hyp_sent_split_remove(self, can):

		can_sent=[[tmp.rstrip('.'or'?'or'!'or','or'\n')] for tmp in sent_tokenize(can, self.lang)]
		return can_sent

	def _ref_sent_split_remove(self, ref):

		ref_sent=[sent_tokenize(tmp,self.lang) for tmp in ref]
		ref_sent_c=[]
		for tmp in ref_sent:
			ref_sent_in=[]
			for tmp2 in tmp:
				ref_sent_in.append(word_tokenize(tmp2.rstrip('.'or'?'or'!'or','or'\n'),self.lang))
			ref_sent_c.append(ref_sent_in)

		return ref_sent_c

	def _token(self, ref_stoken, can, n):

		numer=[]
		ref_len=0

		can=list(ngrams(can,n))

		for tmp in ref_stoken:

			if n==1:
				ref=list(ngrams(tmp,1))
			elif n==2:
				ref=list(ngrams(tmp,2))
			elif n==3:
				ref=list(ngrams(tmp,3))
			else:return 0

			intersect = [t for t in ref if t in can ]
			numer.append(len(intersect))
			ref_len+=len(ref)

		try:
			rec= sum(numer)/ref_len
		except:
			rec=0

		return rec



	def rouge_n(self, ref, can, n=1):

		beta=1
		rec,prec=0,0

		can_sent=self._hyp_sent_split_remove(can)
		can_word=list(itertools.chain(*[word_tokenize(tmp,self.lang) for tmp in can_sent]))
		ref=self._ref_sent_split_remove(ref)

		r_list=[]

		for tmp in ref:
			if n==1:
				r_list.append(self._token(tmp, can_word, 1))
			elif n==2:
				r_list.append(self._token(tmp, can_word, 2))
			elif n==3:
				r_list.append(self._token(tmp, can_word, 3))

		return max(r_list)



	def rouge_l(self, ref, can):

		beta=1
		#check=0

		can= self._hyp_sent_split_remove(can)
		can=[word_tokenize(tmp,self.lang) for tmp in can]
		refs=self._ref_sent_split_remove(ref)

		can_word=list(itertools.chain(*can))

		result_list=[]

		for ref in refs:
			lcs_list=[]
			for ri in ref:
				ri_C=[]
				for ci in can:
					temp=self._lcs(ci,ri)
					ri_C.append(temp)

				ri_C=list(itertools.chain(*ri_C))
				ri_C=set(ri_C)
				lcs_list.append(len(ri_C))

			ref_word=list(itertools.chain(*ref))

			R_lcs=sum(lcs_list)/len(ref_word)
			P_lcs=sum(lcs_list)/len(can_word)

			try:
				F_lcs= (2*R_lcs*P_lcs)/(R_lcs+P_lcs)
			except:
				F_lcs=0
			result_list.append(F_lcs)

		return max(result_list)



	def _lcs(self, can, ref):


		s1=can
		s2=ref
		check=0

		if len(s1)<=len(s2):
			temp=s1
			s1=s2
			s2=temp
			check=1

		m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]

		for x in range(1, 1 + len(s1)):
			for y in range(1, 1 + len(s2)):
				if s1[x - 1] == s2[y - 1]:
					m[x][y] = m[x - 1][y - 1] +1
				else:
					m[x][y]=max(m[x][y-1],m[x-1][y])
		f_x=len(s2)+1
		lcs=m[len(s1)][len(s2)]
		temp=[]


		i=len(s1)
		j=len(s2)

		while m[i][j]!=0:
			if(m[i][j]==m[i][j-1]):
				j-=1
			elif (m[i][j]==m[i-1][j]):
				i-=1
			else:
				if check==0:
					temp.append(s1[i-1])
				if check==1:
					temp.append(s2[j-1])
				i-=1
				j-=1

		return temp
		'''
		for y in reversed(range(1,1+len(s1))):
			for x in reversed(range(1,1+len(s2))):
				if (m[y][x]-m[y-1][x-1]==1) and (m[y][x]-m[y-1][x]==1) and (m[y][x]-m[y][x-1]==1):
					if (y==len(s1)+1) and (x==len(s2)):
						temp.append(x)
					else:
						temp.append(x-1)

		print('the police 만  나와줘야',temp)
		if check==0:
			word=s1
		elif check==1:
			word=s2

		ret_list=[]

		for tmp in range(len(temp)):
			ret_list.append(word[temp[tmp]])

		return ret_list
		'''


	def _skip_bigrams(self, ref_stoken, can_sent, can, n=1):

		beta=1
		numer=[]
		ref_len=0

		candidate=list(skipgrams(can,2,n))
		can_sent=[word_tokenize(tmp,self.lang) for tmp in can_sent]
		can_sk_len=0

		for tmp in ref_stoken:
			ref=list(skipgrams(tmp,2,n))
			intersect=[t for t in ref if t in candidate]
			numer.append(len(intersect))
			ref_len+=len(ref)

		for tmp in can_sent:
			can_sk_len+=len(list(skipgrams(tmp,2,n)))

		prec=sum(numer)/can_sk_len
		rec=sum(numer)/ref_len

		if(prec!=0 and rec!=0):
			score = ((1 + beta**2)*prec*rec)/float(rec + beta**2*prec)
		else:
			score = 0.0
		return score


	def rouge_s(self, ref, can, n):

		can_sent= self._hyp_sent_split_remove(can)
		can_word=list(itertools.chain(*[word_tokenize(tmp,self.lang) for tmp in can_sent]))
		ref= self._ref_sent_split_remove(ref)


		r_list=[]

		for tmp in ref:
			#tmp=list(itertools.chain(*tmp))
			r_list.append(self._skip_bigrams(tmp,can_sent,can_word,n))

		return max(r_list)


	def cider(self, ref, hyp):

		ref_dict=dict()
		hyp_dict=dict()

		ref_dict[0]=ref
		hyp_dict[0]=hyp

		cider_score=Cider()
		score=cider_score.compute_score(ref_dict,hyp_dict)

		return float(score)

	def _process_espresso_output_format(self, result_list):
		temp_list = []
		for k in result_list:
			k = k.split('_')
			if k[1] == 'SP' or k[1] == 'SY':
				continue
			temp_list.append(k)
		return temp_list

	def _generate_enum(self, ref, hyp):
		result_hyp = []
		result_ref = []
		for h in hyp:
			enum_hyp_list = list(enumerate(h))
			result_hyp.append(enum_hyp_list)
		for r in ref:
			enum_ref_list = list(enumerate(r))
			result_ref.append(enum_ref_list)
		return result_hyp, result_ref

	def _tag_pos_meteor(self, sent_list):
		result_list = list()
		for sent in sent_list:
			tagged_sent = EspressoTagger(task='pos').tag(sent)
			tagged_sent = self._process_espresso_output_format(tagged_sent)
			result_list.append(tagged_sent)
		return result_list

	def _match_enums(self,
		enum_hypothesis_list: List[Tuple[int, str]],
		enum_reference_list: List[Tuple[int, str]],
	) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
		"""
		matches exact words in hypothesis and reference and returns
		a word mapping between enum_hypothesis_list and enum_reference_list
		based on the enumerated word id.

		:param enum_hypothesis_list: enumerated hypothesis list
		:param enum_reference_list: enumerated reference list
		:return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
				enumerated unmatched reference tuples
		"""
		word_match = []
		# print("test 213" , enum_hypothesis_list)
		# print("test 124" , enum_reference_list)
		for i in range(len(enum_hypothesis_list))[::-1]:
			for j in range(len(enum_reference_list))[::-1]:
				# print(f"\n \t {enum_hypothesis_list[i][1]} \t {enum_reference_list[j][1]}")
				if enum_hypothesis_list[i][1] == enum_reference_list[j][1]:

					# print("Check!!")
					word_match.append(
						(enum_hypothesis_list[i][0], enum_reference_list[j][0])
					)
					enum_hypothesis_list.pop(i)
					enum_reference_list.pop(j)
					break
		return word_match, enum_hypothesis_list, enum_reference_list


	def _count_chunks(self, matches: List[Tuple[int, int]]) -> int:
		"""
		Counts the fewest possible number of chunks such that matched unigrams
		of each chunk are adjacent to each other. This is used to calculate the
		fragmentation part of the metric.

		:param matches: list containing a mapping of matched words (output of align_words)
		:return: Number of chunks a sentence is divided into post alignment
		"""
		i = 0
		chunks = 1
		while i < len(matches) - 1:
			if (matches[i + 1][0] == matches[i][0] + 1) and (
				matches[i + 1][1] == matches[i][1] + 1
			):
				i += 1
				continue
			i += 1
			chunks += 1
		return chunks

	def _match_syn_with_sejong(self, hyp_list, ref_list):
		syn_match = []
		for i in range(len(hyp_list))[::-1]:
			temp_syn_list = []
			#print("test 344434: ", hyp_list[i])
			if hyp_list[i][1][1] not in self.skip_pos:
				entrys = ssem.entrys(hyp_list[i][1][0])
				for entry in entrys:
					for sense in entry.senses():
						if sense.syn():
							temp_syn_list.append(sense.syn())
			if temp_syn_list:
				hyp_list[i][1].append(deepcopy(temp_syn_list))

			for j in range(len(ref_list))[::-1]:
				is_break = False
				if len(hyp_list[i][1]) == 3:
					for syn in hyp_list[i][1][2]:

						if syn[0] == ref_list[j][1][0]:
							syn_match.append(
								(hyp_list[i][0], ref_list[j][0])
							)
							is_break = True
							hyp_list.pop(i)
							ref_list.pop(j)
							break
				else:
					if hyp_list[i][1] == ref_list[1][1]:
						syn_match.append(
							(hyp_list[i][0], ref_list[j][0])
						)
						is_break = True
						hyp_list.pop(i)
						ref_list.pop(j)
				if is_break:
					break



			# print("test 231232 ", hyp_list[i])


		return syn_match, hyp_list, ref_list

	def meteor(self, ref, hyp):
		ref_tag = self._tag_pos_meteor(ref)
		hyp_tag = self._tag_pos_meteor(hyp)
		meteors = []
		alpha = 0.9
		beta = 3.0
		gamma = 0.5
		enum_hyp, enum_ref = self._generate_enum(ref_tag, hyp_tag)
		# print("test 13333 ", enum_hyp)
		for reference in enum_ref:
			hyp_len = len(enum_hyp[0])
			ref_len = len(reference)

			# 단어/어간 매칭
			word_match, enum_hyp_list, enum_ref_list = self._match_enums(deepcopy(enum_hyp[0]), reference)
			syn_match, enum_hyp_list, enum_ref_list = self._match_syn_with_sejong(enum_hyp_list, enum_ref_list)
			# print("test 123344 " ,enum_ref_list) ## [(0, ['오늘', 'NN']), (6, ['이', 'VB']), (7, ['었다', 'EE'])]

			final_match = sorted(word_match + syn_match)

			#최종 결과 계산
			final_match_count = len(final_match)


			precision = float(final_match_count) / hyp_len
			recall = float(final_match_count) / ref_len
			fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
			chunk_count = float(self._count_chunks(final_match))
			frag = 0.0
			if final_match_count != 0:
				frag = chunk_count / final_match_count
			else:
				frag = 0.0
			penalty = gamma * frag ** beta
			meteors.append((1 - penalty) * fmean)

			# print(word_match)

		return max(meteors)


if __name__=="__main__":
	hyp='봉준호 감독이 아카데미에서 국제영화상을 수상했다.'
	ref=['봉준호가 아카데미에서 각본상을 탔다.']
	metric = StringMetric()
	re = metric.meteor(ref, hyp)
	print(re)
