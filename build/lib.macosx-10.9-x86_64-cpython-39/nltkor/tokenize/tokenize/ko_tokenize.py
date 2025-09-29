# Natural Language Toolkit for Korean: NLTKor's very own tokenizer.
#
# Copyright (C) 2001-2020 NLTKor Project
# Author:
# URL: <http://>
# For license information, see LICENSE.TXT


import re

#for Korean

class Enum(object):
    def __init__(self, names):
        for value, name in enumerate(names.split()): setattr(self, name, value)

class Ko_tokenize():

    def word(target, encoding='utf8'):
        """ Word Tokenizer

        단어 단위로 Tokenizing 한다.

        인자값 목록 (모든 변수가 반드시 필요):

        target : Tokenizing 하고자 하는 대상 문자열

        결과값 : 토크나이징 결과를 list 자료형으로 넘김

        """
        isHangulSyllables = lambda x: unicodedata.name(x).find("HANGUL SYLLABLE") == 0
        isHanjaSyllables = lambda x: unicodedata.name(x).find("CJK") == 0
        isNumber = lambda x: unicodedata.name(x).find("FULLWIDTH DIGIT") == 0 or unicodedata.name(x).find("DIGIT") == 0
        isAlphabet = lambda x: unicodedata.name(x).find("FULLWIDTH LATIN") == 0 or unicodedata.name(x).find("LATIN") == 0
        isAlphabet_Connection = lambda x: x in (".", "-", "_", "|")
        isNumber_Connection = lambda x: x in (".", ",")
        isPunctuation = lambda x: unicodedata.category(x)[0] == "P"
        isSymbol = lambda x: unicodedata.category(x)[0] == "S"
        getCategory = lambda x: unicodedata.category(x)

        TYPE = Enum("UNKNOWN SYMBOL NUMBER PUNCTUATION ALPHABET HANJA HANGUL")

        buf = str()
        type_prev = 0
        type_cur = 0

        if type(target) == str:
            target = target

        for i in range(len(target)):
            ch = target[i]
            ca = str()
            try:
                if isHangulSyllables(ch): type_cur = TYPE.HANGUL
                elif isHanjaSyllables(ch): type_cur = TYPE.HANJA
                elif isNumber(ch): type_cur = TYPE.NUMBER
                elif isAlphabet(ch): type_cur = TYPE.ALPHABET
                elif isAlphabet_Connection(ch) and type_prev == TYPE.ALPHABET:
                    if i+1 < len(target) and not isAlphabet(target[i+1]): type_cur = TYPE.SYMBOL
                    else: type_cur = TYPE.ALPHABET
                elif isNumber_Connection(ch) and type_prev == TYPE.NUMBER:
                    if i+1 < len(target) and not isNumber(target[i+1]): type_cur = TYPE.SYMBOL
                    elif i+1 == len(target): type_cur = TYPE.SYMBOL
                    else: type_cur = TYPE.NUMBER
                elif isPunctuation(ch): type_cur = TYPE.PUNCTUATION
                elif isSymbol(ch): type_cur = TYPE.SYMBOL
                else: type_cur = TYPE.UNKNOWN
                ca = getCategory(ch)
            except:
                type_cur = TYPE.UNKNOWN
            if type_cur == TYPE.PUNCTUATION:
                if ca in ("Ps", "Pe"): buf += " "
                elif i >= 0 and i < len(target) and target[i-1] != target[i]: buf += " "
            elif type_cur != type_prev: buf += " "
            buf += ch
            type_prev = type_cur
        return buf.split()



    def syllable(text,blank=False):
        """
        음절 토크나이저

        음절단위로 tokenizing

        박찬양
        """
        emjeol_list = list()
        for emjeol in text:

          if blank and (emjeol not in ['\n']):
           emjeol_list.append(emjeol)

          elif emjeol not in [' ', '\n']:
           emjeol_list.append(emjeol)

        return emjeol_list


    def sentence(text):
        """
        문장 토크나이저

        문장단위로 tokenizing
        """
        txt=text.replace("\n"," ")
        p=re.compile(r'(?<!\w\.\w.)(?<=\.|\?|\!)\s').split(txt)
        result=[]
        for tmp in p:
            if (tmp == ' ' or tmp== ''):
                continue
            else:result.append(tmp.strip(" "))

        return result
