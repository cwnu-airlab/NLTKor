import re, os, sys
import pandas as pd
import numpy as np
import json
import argparse

class SearchDic :
    def __init__ (self,root) :
        self.root = root

    def build_search_dict(self, word, data) -> dict:
        current_dict = self.root
        _end_word_ = '$$'
        for letter in word:

            current_dict = current_dict.setdefault(letter, {})
        current_dict = current_dict.setdefault(_end_word_, data)
         



    def search_dict(self, word, space_flag=False):
        '''
        TRIE 탐색
        space_flag: if True then including space, otherwise do not including space
        '''

        values = list()
        value_data = list()
        if not word: return self.root.keys()

        current_dict = self.root
        _end_word_ = '$$'
        SPACE = ' '
        s = 0
        for i, letter in enumerate(word):
            #print(i, s, '>', letter, values, value_data, current_dict)
            if letter in current_dict:
                #print('\t', letter, values, value_data, current_dict)
                current_dict = current_dict[letter]
                if _end_word_ in current_dict :
                    values.append(word[s:i+1])
                    value_data.append(current_dict[_end_word_])
            elif space_flag and letter != SPACE and SPACE in current_dict:
                look_ahead_dict = current_dict[SPACE]
                # print('\t==', i, letter, values, look_ahead_dict)
                if letter in look_ahead_dict:
                    current_dict = look_ahead_dict[letter]
            elif space_flag and letter == SPACE:
                # print('\t##', i, letter, word[i+1], values)
                continue
            else:
                # print('\t@@', i, letter, values)
                s = i+1
                current_dict = self.root
        else:
            if values: return values, value_data
            else:	return list(word), value_data


    def save_dict(self, file_path):
        # root dictionary를 pickle 파일로 저장
        with open(file_path, 'wb') as f:
            pickle.dump(self.root, f)
    
    def load_dict(self,file_path) -> dict:
        # pickle 퍄일을 읽어들인다.
        with open(file_path, 'rb') as f:
            return pickle.load(f)
if __name__ == "__main__":
    root = {}
    dict_file = '텍스트파일경로'
    sc = SearchDic(root)
    with open(dict_file, 'r') as f:
        for line in f:
            if ';;' in line[:2]: continue
            k, v = line.strip().split('\t')
            sc.build_search_dict(k, v)
    # print(root)
    word = '고용 노동부'
    values, value_data = sc.search_dict(word, True)
    print(values, value_data)

    word = '2시뉴스외전'
    values, value_data = sc.search_dict( word, True)
    print(values, value_data)
    word = '2시 뉴스외전'
    values, value_data = sc.search_dict( word, True)
    print(values, value_data)

    word = 'gbc'
    values, value_data = sc.search_dict( word, True)
    print(values, value_data)
    
    
