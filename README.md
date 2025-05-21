# Manual [NLTKor]

## **Revision History**

| 번호 | 날짜      | 작성자 | 내용             |
| ---- | --------- | ------ | ---------------- |
| 1    | 2024.4.1  | 김도원 | NLTKo 1.0.0 공개<br> string2string 기능 추가, METEOR Score 추가 |
| 2    | 2024.5.22 | 차정원 | NLTKo 1.1.0 공개 |
| 3    | 2025.2.5 | 이예나 | NLTKor 1.2.0 공개<br> bleu tensor 추가, entment 추가, accurancy norm 추가 |
| 4    | 2025.4.3 | 이예나 | NLTKor 1.2.10 업데이트<br> espresso 오류 수정 |
| 5    | 2025.5.21 | 정찬혁 | NLTKor 1.2.18 업데이트<br> TRIE 검색 추가|




<div style="page-break-after: always;"></div>

## 목차

- [1. 목적](#1.-목적)
- [2. 사용 환경](#2-사용-환경)
  - [2.1 라이브러리 설치](#21-라이브러리-설치)
  - [2.1.1 설치 도중 오류 발생시 해결 방법](#211-설치-도중-오류-발생시-해결-방법)
- [3. 함수 설명](#3-함수-설명)
  - [3.1 토크나이저](#21-토크나이저)
    - [3.1.1 word(text)](#311-wordtext)
    - [3.1.2 sentence(text)](#312-sentencetext)
    - [3.1.3 syllable(text, blank=False)](#313-syllabletext-blankfalse)
- [4. 한국어 문자 처리 함수들](#4-한국어-문자-처리-함수들)
  - [4.1 is_kor_char(ch)](#41-is_kor_charch)
  - [4.2 split_syllable(ch)](#42-split_syllablech)
  - [4.3 join_syllable(Ch1, Ch2, Ch3)](#43-join_syllablech1-ch2-ch3)
  - [4.4 kor_cmp(Ch1, Ch2)](#44-kor_cmpch1-ch2)
  - [4.5 is_kor_syllable(Ch)](#45-is_kor_syllablech)
  - [4.6 is_hanja(Ch)](#46-is_hanjach)
  - [4.7 is_eng_char(Ch)](#47-is_eng_charch)
  - [4.8 is_symbol(Ch)](#48-is_symbolch)
  - [4.9 is_punctuation(Ch)](#49-is_punctuationch)
  - [4.10 is_engConnection(Ch)](#410-is_engconnectionch)
  - [4.11 is_numConnection(Ch)](#411-is_numconnectionch)
- [5. 기본 평가 함수들](#5-기본-평가-함수들)
  - [5.1 Accuracy](#51-accuracy)
  - [5.2 Accurancy norm](#52-accuracy-norm)
  - [5.3 Precision](#53-precision)
  - [5.4 Recall](#54-recall)
  - [5.5 F1 score](#55-f1-score)
  - [5.6 P@k (Precision at k), R@k (Recall ar k)](#56-pk-precision-at-k-rk-recall-ar-k)
  - [5.7 Hit rate @ k](#57-hit-rate--k)
  - [5.8 세종형식 품사태깅 결과 평가](#58-세종형식-품사태깅-결과-평가)
  - [5.9 WER/CER](#59-wercer)
  - [5.10 BLEU](#510-bleu)
  - [5.10.1 BLEU for tensor](#5101-bleu-for-tensor)
  - [5.11 ROUGE](#511-rouge)
  - [5.12 CIDER](#512-cider)
  - [5.13 METEOR](#513-meteor)
  - [5.14 EMR(Entity Mention Recall)](#514-emrentity-mention-recall)
- [6 확장 평가 함수](#6-확장-평가-함수)
  - [6.1 MAUVE](#61-mauve)
  - [6.2 BERT Score](#62-bert-score)
  - [6.3 BART Score](#63-bart-score)
- [7. 한국어 분석 함수 (Espresso)](#7-한국어-분석-함수-espresso)
  - [7.1 품사 태깅](#71-품사-태깅)
  - [7.2 dependency parse](#72-dependency-parse)
  - [7.3 wsd tag](#73-wsd-tag)
  - [7.4 ner tag](#74-ner-tag)
  - [7.5 srl tag](#75-srl-tag)
- [8. 번역 함수](#8-번역-함수)
- [9. 정렬 (Alignment)](#9-정렬-alignment)
  - [9.1 Needleman-Wunsch 알고리즘](#91-needleman-wunsch-알고리즘)
  - [9.2 Hirschberg 알고리즘](#92-hirschberg-알고리즘)
  - [9.3 Smith-Waterman 알고리즘](#93-smith-waterman-알고리즘)
  - [9.4 DTW](#94-dtw)
  - [9.5 Longest Common Subsequence](#95-longest-common-subsequence)
  - [9.6 Longest Common Substring](#96-longest-common-substring)
- [10. 거리 (distance)](#10-거리distance)
    - [10.1 Levenshtein Edit Distance](#101-levenshtein-edit-distance)
    - [10.2 Hamming Distance](#102-hamming-distance)
    - [10.3 Damerau-Levenshtein Distance](#103-damereau-levenshtein-distance)
    - [10.4 Wasserstein Distance](#104-wasserstein-distance)
    - Kullback-Leibler Distance
    - Wasserstein Distance
    - Jensen-Shannon Distance
- [11. 유사도 (similarity)](#11-유사도-similarity)
  - [11.1 코사인 유사도 (Cosine Similarity)](#111-코사인-유사도-cosine-similarity)
  - [11.2 LCSubstring Similarity](#112-lcsubstring-similarity)
  - [11.3 LCSubsequence Similarity](#113-lcsubsequence-similarity)
  - [11.4 Jaro Similarity](#114-jaro-similarity)
- [12. 검색 (search)](#12-검색-search)
  - [12.1 Naive Search](#121-navie-search)
  - [12.2 Rabin-Karp 검색](#122-rabin-karp-검색)
  - [12.3 KMP 검색 알고리즘](#123-kmp-검색)
  - [12.4 Boyer-Moore 검색 알고리즘](#124-boyer-moore-검색)
  - [12.5 Faiss-Semantic 검색](#125-faiss-semantic-검색)
  - [12.6 TRIE 검색](#126-trie-검색)
- [13. 세종전자사전 (ssem)](#13-세종전자사전-ssem)
  - [13.1 객체 확인 방법](#131-객체-확인-방법)
  - [13.2 entry 접근법](#132-entry-접근법)
  - [13.3 sense 접근법](#133-sense-접근법)
  - [13.4 entry 함수 사용법 & 결과](#134-entry-함수-사용법--결과)
  - [13.5 sense 함수 사용법 & 결과](#135-sense-함수-사용법--결과)
- [14. etc](#14-etc)
  - [14.1](#141)


<div style="page-break-after: always;"></div>

## 1. 목적

NLTKor는 한국어를 위한 NLTK이며 기존의 영어에서 사용하는 WordNet 대신 세종전자사전을 사용한다. NLTK와 동일한 함수명을 사용하여 한국어 정보들을 손쉽게 사용하는데 목적이 있다. 설치 방법은 [다음](#21-라이브러리-설치)와 같다.

## 2. 사용 환경

- 운영체제 : ubuntu 18.04, ubuntu 22.04, MacOS
- 언어 : `python3.8`, `python3.9`, `python3.10`, `python3.11`
- 라이브러리 : nltk>=1.1.3, numpy==1.23, faiss-cpu=1.7.3 **※ 해당 NLTKor는 영어 NLTK를 별도로 인스톨해야 함.**

**주의사항**

- Espresso5의 EspressoTagger의 사용가능 환경은 다음과 같다.

| OS     | python                                    | 아키텍처      |
| ------ | ----------------------------------------- | ------------- |
| Mac    | python3.8                                 | arm64         |
| ubuntu | python3.8 python3.9 python3.10 python3.11 | arm64, x86_64 |

### 2.1 라이브러리 설치

해당 라이브러리를 설치하기 위해서 아래와 동일하게 명령어 라인에서 입력한다.

```h
$ pip install nltkor

```

##### 2.1.1. 설치 도중 오류 발생시 해결 방법

- 만약 ubuntu에서 `pip install`을 진행하였을 때, 오류가 발생하여 제대로 설치가 되지않는다면, 아래의 명령어들을 실행하여 해결할 수 있다.

```h
apt update
apt-get install g++
apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-cache policy python3.8
apt install python3.8
apt install python3.8-dev
apt-get install python3.8-distutils
apt install git
```

- `apt install pythonx.x-dev` (x.x는 사용자의 python 버전에 맞게 입력)
- `apt-get install pythonx.x-distutils` (x.x는 사용자의 python 버전에 맞게 입력)

- 만약 설치 도중 해당 에러가 발생한다면
```h
× Getting requirements to build wheel did not run successfully.
│ exit code: 1
╰─> See above for output.
```

```h
pip install wheel
```
로 wheel을 먼저 설치해야한다.

<div style="page-break-after: always;"></div>

## 3. 함수 설명

### 3.1 토크나이저 (tokenizer)

nltkor의 sent_tokenize(), word_tokenize() 사용 방법과 동일하게 사용가능하다.

#### 3.1.1 word(text)

단어 단위로 Tokenizing 한다.

```
text : Tokenizing 하고자 하는 대상 문자열

결과값 : 토크나이징 결과를 list 자료형으로 넘김
```

사용예시

```
>>> from nltkor.tokenize import Ko_tokenize
>>> txt = "안녕하세요. 저는 키딩입니다. 창원대학교에 재학중입니다."
>>> Ko_tokenize.word(txt)
['안녕하세요.', '저는', '키딩입니다.', '창원대학교에', '재학중입니다.']
```

#### 3.1.2 sentence(text)

문장 단위로 Tokenizing 한다.

```
text : Tokenizing 하고자 하는 대상 문자열들

결과값 : 토크나이징 결과를 list 자료형으로 넘김
```

사용예시

```
>>> from nltkor.tokenize import Ko_tokenize
>>> txt = "안녕하세요. 저는 키딩입니다. 창원대학교에 재학중입니다."
>>> Ko_tokenize.sentence(txt)
['안녕하세요.', '저는 키딩입니다.', '창원대학교에 재학중입니다.']
```

#### 3.1.3 syllable(text, blank=False)

음절 단위로 Tokenizing 한다.

```
text : Tokenizing 하고자 하는 대상 문자열들
blank : True: 공백을 살린다.
        False: 공백을 무시한다.

결과값 : 토크나이징 결과를 list 자료형으로 넘김
```

사용예시

```
>>> from nltkor.tokenize import Ko_tokenize
>>> txt = "안녕하세요. 저는 키딩입니다. 창원대학교에 재학중입니다."
>>> Ko_tokenize.syllable(txt)
['안', '녕', '하', '세', '요', '.', '저', '는', '키', '딩', '입', '니', '다', '.', '창', '원', '대', '학', '교', '에', '재', '학', '중', '입', '니', '다', '.']
>>> Ko_tokenize.syllable(txt, True)
['안', '녕', '하', '세', '요', '.', ' ', '저', '는', ' ', '키', '딩', '입', '니', '다', '.', ' ', '창', '원', '대', '학', '교', '에', ' ', '재', '학', '중', '입', '니', '다', '.']
```

<div style="page-break-after: always;"></div>

### 4. 한국어 문자 처리 함수들

#### 4.1 is_kor_char(Ch)

문자가 한국어 문자인지 검사한다.

```
Ch : 문자

결과값 : True, False
```

사용예시

```
>>> from nltkor import Kor_char
>>> print(Kor_char.is_kor_char('ㄱ'))
True
>>> print(Kor_char.is_kor_char('A'))
False
```

#### 4.2 split_syllable(Ch)

음절을 '초성/중성/종성'으로 분리한다.

```
Ch : 음절

결과값 : ('초성', '중성', '종성')
```

사용예시

```
>>> from nltkor import Kor_char
>>> print(Kor_char.split_syllable('감'))
('ㄱ', 'ㅏ', 'ㅁ')
>>> print(Kor_char.split_syllable('가'))
('ㄱ', 'ㅏ', '')
```

#### 4.3 join_syllable(Ch1, Ch2, Ch3)

'초성/중성/종성'이 주어지면 음절로 조합한다.

```
Ch1 : 초성
Ch2 : 중성
Ch3 : 종성

결과값 : 음절
```

사용예시

```python
>>> from nltkor import Kor_char
>>> print(Kor_char.join_syllable('ㄱ', 'ㅏ', 'ㅁ'))
감'
```

#### 4.4 kor_cmp(Ch1, Ch2)

한글 문자 두 개를 비교한다. 문자열은 사용하지 못한다.

```
Ch1 : 문자
Ch2 : 문자

결과값 : -1 : 앞 문자가 순서가 먼저인 경우
        0 : 두 문자가 같은 경우
        1 : 앞 문자가 순서가 뒤에 있는 경        
```

사용예시

```python
>>> from nltkor import Kor_char
>>> print(Kor_char.kor_cmp('가', '라'))
-1
>>> print(Kor_char.kor_cmp('가', '가'))
0
>>> print(Kor_char.kor_cmp('라', '가'))
1
```

#### 4.5 is_kor_syllable(Ch)

한글 음절인지 판별한다.

```
Ch : 음절

결과값 : True, False        
```

사용예시

```python
>>> from nltkor import Kor_char
>>> print(Kor_char.is_kor_syllable('라'))
True
>>> print(Kor_char.is_kor_syllable('ㄹ'))
False
>>> print(Kor_char.is_kor_syllable('a'))
False
```

#### 4.6 is_hanja(Ch)

한자인지 판별한다.

```
Ch : 문

결과값 : True, False        
```

사용예시

```python
>>> from nltkor import Kor_char
>>> print(Kor_char.is_hanja('成'))
True
>>> print(Kor_char.is_hanja('한'))
False
>>> print(Kor_char.is_hanja('!'))
False
>>> print(Kor_char.is_hanja('a'))
False
```

#### 4.7 is_eng_char(Ch)

영문자인지 판별한다.

```
Ch : 문자

결과값 : True, False        
```

사용예시

```python
>>> from nltkor import Kor_char
>>> print(Kor_char.is_eng_char('a'))
True
>>> print(Kor_char.is_eng_char('가'))
False
>>> print(Kor_char.is_eng_char('A'))
True
```

#### 4.8 is_symbol(Ch)

기호인지 판별한다. '@', '#', '%', '&', '\*', '!', '.', '?'는 punctutation으로 인식한다.

```
Ch : 문자

결과값 : True, False        
```

사용예시

```python
>>> from nltkor import Kor_char
>>> print(Kor_char.is_symbol('★'))
True
>>> print(Kor_char.is_symbol('가'))
False
>>> print(Kor_char.is_symbol('a'))
False
>>> print(Kor_char.is_symbol('@'))
False
>>> print(Kor_char.is_symbol('$'))
True
>>> print(Kor_char.is_symbol('@'))
False
>>> print(Kor_char.is_symbol('!'))
False
>>> print(Kor_char.is_symbol('%'))
False
>>> print(Kor_char.is_symbol('^'))
True
>>> print(Kor_char.is_symbol('&'))
False
>>> print(Kor_char.is_symbol('*'))
False
>>> print(Kor_char.is_symbol('~'))
True
```

#### 4.9 is_punctuation(Ch)

입력 문자가 구두점인지 확인한다.

```
Ch : 문자

결과값 : True, False        
```

사용예시

```python
>>> from nltkor import Kor_char
>>> print(Kor_char.is_punctuation('.'))
True
>>> print(Kor_char.is_punctuation('?'))
True
>>> print(Kor_char.is_punctuation('!'))
True
>>> print(Kor_char.is_punctuation('...'))
function expect a character, check the value
None
>>> print(Kor_char.is_punctuation('@'))
True
>>> print(Kor_char.is_punctuation('%'))
True
>>> print(Kor_char.is_punctuation('&'))
True
>>> print(Kor_char.is_punctuation('*'))
True
>>> print(Kor_char.is_punctuation('~'))
False
```

#### 4.10 is_engConnection(Ch)

입력 문자가 영어 연결 문자인지 판별한다.

```
Ch : 문자

결과값 : True, False        
```

사용예시

```python
>>> from nltkor import Kor_char
>>> print(Kor_char.is_engConnection('.'))
True
>>> print(Kor_char.is_engConnection('-'))
True
>>> print(Kor_char.is_engConnection('_'))
True
>>> print(Kor_char.is_engConnection('|'))
True
>>> print(Kor_char.is_engConnection('='))
False
```

#### 4.11 is_numConnection(Ch)

입력 문자가 수 연결 문자인지 판별한다.

```
Ch : 문자

결과값 : True, False        
```

사용예시

```python
>>> from nltkor import Kor_char
>>> print(Kor_char.is_numConnection(','))
True
>>> print(Kor_char.is_numConnection('.'))
True
>>> print(Kor_char.is_numConnection('-'))
False
```

<div style="page-break-after: always;"></div>

### 5. 기본 평가 함수들

평가 함수들은 기본 언어가 한국어로 설정되어 있다. 이것을 영어로 변경하려면 `DefaultMetric(lang='en')` 로 생성자를 변경해야 한다.

#### 5.1 Accuracy

```python
>>> from nltkor.metrics import DefaultMetric
>>> y_true = ['a', 'b', 'b', 'd', 'a']
>>> y_pred = ['a', 'c', 'b', 'a', 'b']
>>> DefaultMetric().accuracy_score(y_true,y_pred)
0.4
```

#### 5.2 Accuracy norm
일반적으로 생성형 모델의 예측 정확도를 정규화(normalization)한 값을 의미한다.
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import random
from nltkor.metrics import DefaultMetric

# 모델과 토크나이저 로드
model_name = 'gpt2-medium'
model = AutoModelForCausalLM.from_pretrained(model_name).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# HellaSwag 형식의 임의 데이터 생성
examples = [
    {
        "input_text": "The boy threw a frisbee and",
        "candidates": ["it landed on the roof.", "the dog chased it.", "it broke.", "it disappeared."],
        "label": 1
    },
    {
        "input_text": "The man started cooking by",
        "candidates": ["turning on the stove.", "reading a book.", "watching TV.", "taking a nap."],
        "label": 0
    },
    {
        "input_text": "The girl put on her shoes and",
        "candidates": ["went for a run.", "started sleeping.", "ate breakfast.", "watched TV."],
        "label": 0
    },
    {
        "input_text": "The cat jumped onto the table and",
        "candidates": ["fell asleep.", "knocked over a glass.", "started barking.", "flew away."],
        "label": 1
    },
    {
        "input_text": "The boy opened the door and saw",
        "candidates": ["a pizza delivery.", "nothing.", "a spaceship.", "an elephant."],
        "label": 0
    },
    {
        "input_text": "The woman picked up her phone and",
        "candidates": ["called her friend.", "ate it.", "threw it away.", "danced."],
        "label": 0
    },
    {
        "input_text": "After finishing his homework, the student",
        "candidates": ["went to sleep.", "built a house.", "flew a kite.", "started driving."],
        "label": 0
    },
    {
        "input_text": "The dog found a stick and",
        "candidates": ["ran away with it.", "ate the sofa.", "drove a car.", "built a boat."],
        "label": 0
    },
    {
        "input_text": "She turned off the lights and",
        "candidates": ["went to bed.", "started cooking.", "went jogging.", "washed her car."],
        "label": 0
    },
    {
        "input_text": "The chef tasted the soup and",
        "candidates": ["smiled.", "cried loudly.", "threw it away.", "started singing."],
        "label": 0
    }
]

# 평가 실행 및 결과 출력
correct = 0
memory_usages = []
inference_times = []

for example in examples:
    cor, metrics = DefaultMetric().accuracy_norm(model, tokenizer, example["input_text"], example["candidates"], example["label"])
    correct += cor
    memory_usages.append(metrics["reserved_memory"])
    inference_times.append(metrics["inference_time"])

accuracy = correct / len(examples)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Time: {sum(inference_times)/len(inference_times)}, memory: {sum(memory_usages)/len(memory_usages)}")
```
**결과**
```
Accuracy: 20.00
Time: 0.05374705195426941, memory: 1409.9
```

#### 5.3 Precision

'micro' precision은 'Accuracy'와 같이 $2/5 = 0.4$ 로 계산이 된다.
'macro' precision은 각 클래스별로 precision을 계산하여 평균을 낸다. 다음 예에서는 'a'=1/2, 'b' = 1/2, 'c' = 0, 'd' = 0. 따라서 1/4 = 0.25가 된다.

```python
>>> from nltkor.metrics import DefaultMetric
>>> y_true = ['a', 'b', 'b', 'd', 'a']
>>> y_pred = ['a', 'c', 'b', 'a', 'b']
>>> DefaultMetric().precision_score(y_true, y_pred,'micro')
0.4
>>> DefaultMetric().precision_score(y_true, y_pred,'macro')
0.25
```

#### 5.4 Recall

```python
>>> from nltkor.metrics import DefaultMetric
>>> y_true = ['a', 'b', 'b', 'd', 'a']
>>> y_pred = ['a', 'c', 'b', 'a', 'b']
>>> DefaultMetric().recall_score(y_true,y_pred,'micro')
0.4
>>> DefaultMetric().recall_score(y_true,y_pred,'macro')
0.25
```

#### 5.5 F1 score

```python
>>> from nltkor.metrics import DefaultMetric
>>> y_true = ['a', 'b', 'b', 'd', 'a']
>>> y_pred = ['a', 'c', 'b', 'a', 'b']
>>> DefaultMetric().f1_score(y_true,y_pred,'micro')
0.4000000000000001
>>> DefaultMetric().f1_score(y_true,y_pred,'macro')
0.25
```

#### 5.6 P@k (Precision at k), R@k (Recall ar k)

```python
>>> from nltkor.metrics import DefaultMetric
>>> y_pred = [5, 2, 4, 1, 3, 2, 5, 6, 7]
>>> y_true = [1, 3, 6, 7, 1, 5]
>>> DefaultMetric().precision_at_k(y_true,  y_pred, 5)
0.8
>>> DefaultMetric().recall_at_k(y_true,y_pred, 5)
0.6666666666666666
```

#### 5.7 Hit rate @ k

'user', 'h_pred'는 정렬된 이중 리스트 형식이다. 다음 예제에서 k = 3이다. 이 경우에 'h_pred[:k]'까지만 평가한다.

```python
>>> from nltkor.metrics import DefaultMetric
>>> user = [[5, 3, 2], [9, 1, 2], [3, 5, 6], [7, 2, 1]]
>>> h_pred = [[15, 6, 21, 3], [15, 77, 23, 14], [51, 23, 21, 2], [53, 2, 1, 5]]
>>> DefaultMetric().hit_rate_at_k(user, h_pred, 3)
0.25
```

#### 5.8 세종형식 품사태깅 결과 평가

다음 예제와 같은 품사 태깅 결과를 입력하여 성능을 측정한다.

```python
'''
우리    우리/NP    우리/NP
나라에    나라/NNG+에/JKB    나라/NNG+에/JKB
있는    있/VA+는/ETM    있/VA+는/ETM
식물의    식물/NNG+의/JKG    식물/NNG+의/JKG
종수만    종/NNG+수/NNG+만/JX    종/ETM+수/NNB+만/JX
하여도    하/VV+여도/EC    하/VV+여도/EC
수천종이나    수천/NR+종/NNG+이나/JX    수천/XSN+종/NNG+이나/JX
되며    되/VV+며/EC    되/VV+며/EC
그중에는    그중/NNG+에/JKB+는/JX    그중/NNG+에/JKB+는/JX
경제적가치가    경제적가치/NNG+가/JKS    경제적가치/NNG+가/JKS
'''
>>> DefaultMetric().pos_eval(test.txt)
'''
입력 텍스트 파일 형식
: 어절    정답    결과

반환 값
:Eojeol Accuracy, Token precision, Token recall, Token f1
:어절 정확도, 토큰 예측율, 토큰 재현율, 토큰 조화평균

'''
(0.8, 0.8636363636363636, 0.8636363636363636, 0.8636363636363636)
```

#### 5.9 WER/CER

- wer (단어 오류율) : 두 입력 문장 사이의 단어 오류율 반환
- cer (음절 오류율) : 두 입력 문장 사이의 문자(음절) 오류율 반환

```
파라미터
    reference : str
    hypothesis : str

결과값
    scores : flaot
```

```python
>>> from nltkor.metrics import DefaultMetric
>>> ref="신세계그룹이 인천 SK와이번스 프로야구단을 인수했다"
>>> hyp="신세계 SK와이번스 프로야구단 인수"
>>> DefaultMetric().wer(ref, hyp)
0.8
>>> DefaultMetric().cer(ref, hyp)
0.3333333333333333
```

#### 5.10 BLEU

- bleu_n : bleu-n(1,2,3,4) 스코어 반환

  각 n만 고려한 스코어

- bleu : bleu score 값 반환 (N=4)

  (1~4)-gram을 모두 고려한 스코어이며 일반적인 의미의 bleu 스코어 (0.25,0.25, 0.25, 0.25)

```
  파라미터
      reference : list of str
      candidate : list
      n : int

  결과값
      scores : flaot
  ::
          candidate=[sentence]

          multi_reference=[
                      ref1 sentence,
                      ref2 sentence,
                      ...]
```

```python
>>> from nltkor.metrics import DefaultMetric
>>> can=['빛을 쐬는 노인은 완벽한 어두운곳에서 잠든 사람과 비교할 때 강박증이 심해질 기회가 훨씬 높았다']
>>> ref=['빛을 쐬는 사람은 완벽한 어둠에서 잠든 사람과 비교할 때 우울증이 심해질 기회가 훨씬 높았다']
>>> DefaultMetric().bleu_n(ref,can,1)
0.7857142857142857
>>> DefaultMetric().bleu_n(ref,can,2)
0.5384615384615384
>>> DefaultMetric().bleu_n(ref,can,3)
0.3333333333333333
>>> DefaultMetric().bleu_n(ref,can,4)
0.18181818181818182
>>> DefaultMetric().bleu(ref,can)
0.4001601601922499
```

#### 5.10.1 BLEU for tensor
- 각 score의 값이 tensor 로 반환한다.
```python
>>> from nltkor.metrics import DefaultMetric
>>> import torch
>>> can=torch.tensor([[1,2,3,4,5],[3,4,5,6,4]])
>>> ref=torch.tensor([[1,2,3,4,5],[3,5,6,7,10]])
>>> DefaultMetric().bleu_tensor(ref,can,1)
tensor(0.8000)
>>> DefaultMetric().bleu_tensor(ref,can,2)
tensor(0.6250)
>>> DefaultMetric().bleu_tensor(ref,can,3)
tensor(0.5000)
>>> DefaultMetric().bleu_tensor(ref,can,4)
tensor(0.5000)
>>> DefaultMetric().bleu_tensor(ref,can)
tensor(0.5946)

```

#### 5.11 ROUGE

※ rouge는 recall based score이며 l, s는 f-measure를 사용하며 n은 recall score이다.

- rouge-n: rouge-n(1,2,3) 스코어 반환

  unigram, bigram, trigram 등 문장 간 중복되는 n-gram을 비교하는 지표

- rouge-l : rouge-lcs 스코어 반환

  LCS를 이용하여 최장길이로 매칭되는 문자열을 측정한다. ROUGE-2와 같이 단어들의 연속적 매칭을 요구하지 않고, 문자열 내에서 발생하는 매칭을 측정하기 때문에 유연한 성능비교가 가능

- rouge-s: rouge-s(1,2,3) 스코어 반환

  Window size 내에 위치하는 단어쌍(skip-gram)들을 묶어 해당 단어쌍들이 얼마나 중복되어 나타나는지 측정

```
파라미터
    reference : list of str(sentences)
    hypothesis: str (sentences)
    n : int

결과값
    rouge score : flaot

::
        hypothesis=hyp summary

        multi_reference=[
                    ref1_summary,
                    ref2_summary,
                    ...]
```

```python
>>> from nltkor.metrics import DefaultMetric
>>> ref_list=["아이폰 앱스토어에 올라와 있는 앱 개발사들은 이제 어느 정보를 가져갈 것인지 정확하게 공지해야함과  동시에 이용자의 승인까지 받아야 합니다."]
>>> hyp="아이폰 앱스토어에 올라와 있는 앱 개발사들은 이제 어느 정보를 가져갈 것인지 공지해야함과 동시에 승인까지 받아야 합니다."
>>> DefaultMetric().rouge_n(ref_list,hyp,1)
0.8888888888888888
>>> DefaultMetric().rouge_n(ref_list,hyp,2)
0.7647058823529411
>>> DefaultMetric().rouge_n(ref_list,hyp,3)
0.625
>>> DefaultMetric().rouge_l(ref_list,hyp)
0.9411764705882353
>>> DefaultMetric().rouge_s(ref_list,hyp,1)
0.8064516129032258
>>> DefaultMetric().rouge_s(ref_list,hyp,2)
0.8222222222222222
>>> DefaultMetric().rouge_s(ref_list,hyp,3)
0.8275862068965517
```

#### 5.12 CIDER

TF-IDF를 n-gram에 대한 가중치로 계산하고 참조 캡션과 생성 캡션에 대한 유사도를 측정

```
파라미터
    reference : list of str(sentences)
    hypothesis: list (sentence)

결과값
    cider score : flaot

::
        hypothesis=[hyp sentence]

        multi_reference=[
                    ref1_sentence,
                    ref2_sentence,
                    ...]

```

```python
>>> from nltkor.metrics import DefaultMetric
>>> ref1=['뿔 달린 소 한마리가 초원 위에 서있다']
>>> ref2=['뿔과 긴 머리로 얼굴을 덮은 소 한마리가 초원 위에 있다']
>>> hyp=['긴 머리를 가진 소가 초원 위에 서있다']
>>> DefaultMetric().cider(ref1, hyp)
0.2404762
>>> DefaultMetric().cider(ref2, hyp)
0.1091321
>>> ref_list=['뿔 달린 소 한마리가 초원 위에 서있다','뿔과 긴 머리로 얼굴을 덮은 소 한마리가 초원 위에 있다']
>>> DefaultMetric().cider(ref_list, hyp)
0.1748041
```

#### 5.13 METEOR

- METEOR (Meter For Evaluation of Translation with Explicit Ordering )

  : unigram precision, recall, f-mean을 기반으로 하며, 어간 (Stemming), 동의어 (synonym)일치를 포함한다.

  : 평가의 기본단위는 문장이다.

- **동의어 추출은 세종의미사전을 활용하였으며 본 라이브러리의 형태소 분석기를 이용하였다.**

```python
>>> from nltkor.metrics import DefaultMetric
>>> ref=['봉준호가 아카데미에서 각본상을 탔다.']
>>> hyp=['봉준호 감독이 아카데미에서 국제영화상을 수상했다.']
>>> DefaultMetric().meteor(ref, hyp)
0.40879120879120884
>>> ref=['오늘 매우 양식이 부족하였다.']
>>> hyp=['현재 식량이 매우 부족하다.']
>>> DefaultMetric().meteor(ref, hyp)
0.5645569620253165
>>> ref=['오늘 양식이 매우 부족하였다.', '오늘 매우 양식이 부족하였다.', '오늘 식량이 매우 부족하였다.']
>>> hyp=['현재 식량이 매우 부족하다.']
>>> DefaultMetric().meteor(ref, hyp)
0.6303797468354431
```

#### 5.14 EMR(Entity Mention Recall)


요약된 텍스트가 참조 문서에 등장하는 중요 개체를 얼마나 잘 유지하고 있는지에 대한 평가 지표이다.

EMR().entity(원본 텍스트,요약된 텍스트)
```python
>>> # -*- coding: utf-8 -*-
>>> from nltkor.metrics import EntMent
>>> EMR().entity("국립창원대학교(총장 박민원)가 사천우주항공캠퍼스 개교와 함께 2025학년도 사천우주항공공학부 입학식을 7일 오전 11시 사천우주항공캠퍼스에서 열었다.이날 행사에는 박민원 총장을 비롯해 국민의힘 서천호 국회의원(사천·남해·하동), 윤영빈 우주항공청장, 박동식 사천시장, 김규헌 사천시의회 의장, 지역 유관기관 관계자들과 신입생 및 가족들이 참석했다. 글로컬대학30사업 선정에 따라 국립창원대와 통합을 추진 중인 경남도립거창대학, 경남도립남해대학 관계자도 함께 자리했다.행사는 1부 현판 제막식과 2부 입학식으로 진행됐으며, 박동식 사천시장은 신입생들에게 축하 선물로 금배지를 전달했고, 박민원 총장은 캠퍼스 설립에 기여한 유공자들에게 표창장을 수여했다.","국립창원대학교는 4월 7일 사천우주항공캠퍼스에서 2025학년도 사천우주항공공학부 입학식을 개최했다. 이날 행사에는 박민원 총장, 서천호 국회의원, 윤영빈 우주항공청장, 박동식 사천시장 등 주요 인사와 신입생 및 가족들이 참석했으며, 글로컬대학30사업과 관련된 거창대학·남해대학 관계자들도 함께했다. 행사는 현판 제막식과 입학식으로 나뉘어 진행되었고, 신입생들에게는 금배지가, 캠퍼스 설립 유공자들에게는 표창장이 수여되었다.")
Downloading Espresso5 model...
0.8888888888888888
```


### 6 확장 평가 함수

#### 6.1 MAUVE

개방형 텍스트 생성의 뉴럴 텍스트와 인간 텍스트 비교 측정 지표이다. <br/><br/>
참고 논문 : https://arxiv.org/abs/2102.01454

**주의 사항**
한국어의 경우 p와 q의 문장 개수가 각각 최소 50개 이상이여야 제대로 된 결과가 나온다.

- **init**(model_name_or_path: str) -> None : 토크나이징과 임베딩을 진행할 모델을 입력받아 Mauve 클래스를 초기화 한다.

  - model은 현재 huggingface에서 제공되는 모델만 사용가능합니다. 로컬 모델은 안됩니다.

- compute(self,

            p_features=None, q_features=None,
            p_tokens=None, q_tokens=None,
            p_text=None, q_text=None,
            num_buckets='auto', pca_max_data=-1, kmeans_explained_var=0.9,
            kmeans_num_redo=5, kmeans_max_iter=500,
            device_id=-1, max_text_length=1024,
            divergence_curve_discretization_size=25, mauve_scaling_factor=5,
            verbose=False, seed=25, batch_size=1, use_float64=False,

  ) -> SimpleNamespace(mauve, frontier_integral, p_hist, q_hist, divergence_curve)

  - `p_features`: (n, d) 모양의 `numpy.ndarray`, 여기서 n은 생성 개수
  - `q_features`: (n, d) 모양의 `numpy.ndarray`, 여기서 n은 생성 개수
  - `p_tokens`: 길이 n의 리스트, 각 항목은 모양 (1, 길이)의 torch.LongTensor
  - `q_tokens`: 길이 n의 리스트, 각 항목은 모양 (1, 길이)의 torch.LongTensor
  - `p_text`: 길이가 n인 리스트, 각 항목은 문자열
  - `q_text`: 길이가 n인 리스트, 각 항목은 문자열
  - `num_buckets`: P와 Q를 양자화할 히스토그램의 크기, Options: `'auto'` (default, n/10를 뜻함) 또는 정수
  - `pca_max_data`: PCA에 사용할 데이터 포인터의 수, `-1`이면 모든 데이터를 사용, Default -1
  - `kmeans_explained_var`: PCA에 의한 차원 축소를 유지하기 위한 데이터 분산의 양, Default 0.9
  - `kmeans_num_redo`: k-평균 클러스터링을 다시 실행하는 횟수(최상의 목표가 유지됨), Default 5
  - `kmeans_max_iter`: k-평균 반복의 최대 횟수, Default 500
  - `device_id`: 기능화를 위한 장치. GPU를 사용하려면 gpu_id(예: 0 또는 3)를 제공, CPU를 사용하려면 -1
  - `max_text_length`: 고려해야 할 최대 토큰 수, Default 1024
  - `divergence_curve_discretization_size`: 발산 곡선에서 고려해야 할 점의 수. Default 25.
  - `mauve_scaling_factor`: 논문의 상수`c` Default 5
  - `verbose`: True인 경우 실행 시간 업데이트를 화면에 출력
  - `seed`: k-평균 클러스터 할당을 초기화하기 위한 무작위 시드
  - `batch_size`: 특징 추출을 위한 배치 크기

  :return

  - `out.mauve`, MAUVE 점수인 0에서 1 사이의 숫자, 값이 높을수록 P가 Q에 더 가깝다는 의미
  - `out.frontier_integral`, 0과 1 사이의 숫자, 값이 낮을수록 P가 Q에 더 가깝다는 것을 의미
  - `out.p_hist`, P에 대해 얻은 히스토그램, `out.q_hist`와 동일
  - `out.divergence_curve`에는 발산 곡선의 점이 포함, 모양은 (m, 2), 여기서 m은 `divergence_curve_discretization_size`

```python
>>> from nltkor.metrics import Mauve
>>> p_ko = ['누나가 동생을 등에 업는다.',
'나는 퀴즈의 정답을 맞혔다.',
'입에 고기를 마구 욱여넣었다.',
'마음이 너무 설렌다.', ...

'봄눈이 녹고 있다.',
'길이 얼어 있다.',
] # total 88

>>> q_ko = ['안녕하세요! 오늘은 어떤 날씨인가요?',
'한국 음식 중 어떤 것이 제일 좋아하세요?',
'학교에서 친구들과 함께 공부하는 게 즐거워요.',
'주말에는 가족과 함께 시간을 보내는 것이 좋아요.', ...

'좋아하는 책을 읽으면서 여유로운 주말을 보내는 것이 행복해요.',
'한국의 다양한 음식을 맛보는 것이 여행의 매력 중 하나에요.',
] # total 88

>>> result = Mauve('skt/kobert-base-v1').compute(p_text=p_ko, q_text=q_ko, device_id = 0, max_text_length=256, verbose=False)
>>> print(result.mauve)
0.6333866808068385
```

#### 6.2 BERT Score

- **init**(model_name_or_path: str | None = None, lang: str | None = None, num_layers: int | None = None, all_layers: bool = False, use_fast_tokenizer: bool = False, device: str = 'cpu', baseline_path: str | None = None) -> None : BERT Score를 초기화하는 생성자입니다.
  - model_name_or_path : BERT 모델의 이름 또는 경로 (huggingface.co에서 가져옵니다.)
  - lang : BERT 모델의 언어 (kor | eng)
  - num_layers : BERT 모델의 레이어 수
  - device : BERT 모델을 실행할 장치 (cpu | cuda)
- compute(source_sentences: List[str], target_sentences: List[str] | List[List[str]], batch_size: int = 4, idf: bool = False, nthreads: int = 4, return_hash: bool = False, rescale_with_baseline: bool = False, verbose: bool = False) -> dict | str | None : 두 문장의 BERT Score를 계산한다.
- 모델은 huggingface.co에서 다운받습니다. (https://huggingface.co/bert-base-uncased)
  - model_name_or_path 파라미터에는 hunggingface.co/ 뒷부분을 넣어줍니다. `model_name_or_path = 'bert-base-uncased'` (https://huggingface.co/<mark>bert-base-uncased</mark>)

```python
>>> from nltkor.metrics import BERTScore
>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result = BERTScore(model_name_or_path='skt/kobert-base-v1', lang='kor', num_layers=12).compute([sent1], [sent2])
>>> print(result)
{'precision': array([0.78243864], dtype=float32), 'recall': array([0.78243864], dtype=float32), 'f1': array([0.78243864], dtype=float32)}
```

#### 6.3 BART Score

- **init**(model_name_or_path='facebook/bart-large-cnn', tokenizer_name_or_path: str | None = None, device: str = 'cpu', max_length=1024) -> None : BART Score를 초기화하는 생성자입니다.
  - model_name_or_path : BART 모델의 이름 또는 경로 (huggingface.co에서 가져옵니다.)
  - device : BART 모델을 실행할 장치 (cpu | cuda)
- compute(source_sentences: List[str], target_sentences: List[str] | List[List[str]], batch_size: int = 4, agg: str = 'mean') -> Dict[str, List[float]] : 두 문장의 BART Score를 계산한다.
- 모델은 huggingface.co에서 다운받습니다. (https://huggingface.co/facebook/bart-large-cnn)
  - model_name_or_path 파라미터에는 hunggingface.co/ 뒷부분을 넣어줍니다. `model_name_or_path = 'facebook/bart-large-cnn'` (https://huggingface.co/<mark>facebook/bart-large-cnn</mark>)

```python
>>> from nltkor.metrics import BARTScore
>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result = BARTScore().compute([sent1], [sent2])
>>> print(result)
{'score': array([-2.97229409])}
```

<div style="page-break-after: always;"></div>

### 7. 한국어 분석 함수 (Espresso)

Espresso5 모델을 사용하여 한국어 분석을 할 수 있다.

- tag(task: str, sentence: str) -> List[str] : 문장의 각 토큰에 대한 task의 태깅 결과를 반환한다.
- 처음 사용하면 다음과 같이 모델을 다운로드 받는 과정이 있다. 모델이 업데이트 되어서 다시 받을려고 한다면 'tag/data' 폴드를 지우고 실행한다.

  ```python
  >>> from nltkor.tag import EspressoTagger
  >>> sent = "나는 아름다운 강산에 살고있다."
  >>> tagger = EspressoTagger(task='pos')
  /나의 실행 경로/nltkor/tag/data.zip
  Downloading Espresso5 model...
  ```

#### 7.1 품사 태깅

한 문장에 대한 '(형태소, 품사)' 결과를 출력한다. 여러 문장을 입력하면 하나의 문장으로 간주하여 처리한다.

```
text : 한 문장을 나타내는 문자열
lemma : True : 원형 복원 (default)
         False : 형태소 분리하지만 원형복원 하지 않
```

사용 예시

```python
>>> from nltkor.tag import EspressoTagger
>>> tagger = EspressoTagger(task='pos')
>>> sent = "우리는 아름다운 강산에 살고 있다."
>>> print(tagger.tag(sent))
[('우리', 'NN'), ('는', 'JJ'), (' ', 'SP'), ('아름답', 'VB'), ('ㄴ', 'EE'), (' ', 'SP'), ('강산', 'NN'), ('에', 'JJ'), (' ', 'SP'), ('살', 'VB'), ('고', 'EE'), (' ', 'SP'), ('있', 'VB'), ('다', 'EE'), ('.', 'SY')]
```

`lemma=False` 파라미터를 주어서 원형 복원을 하지않고 품사 태깅을 할 수 있다.

```python
>>> from nltkor.tag import EspressoTagger
>>> tagger = EspressoTagger(task='pos')
>>> sent = "나는 아름다운 강산에 살고있다."

>>> tagger = EspressoTagger(task='pos')
>>> print(tagger.tag(sent, lemma=False))
[('우리', 'NN'), ('는', 'JJ'), (' ', 'SP'), ('아름다운', 'VB'), (' ', 'SP'), ('강산', 'NN'), ('에', 'JJ'), (' ', 'SP'), ('살', 'VB'), ('고', 'EE'), (' ', 'SP'), ('있', 'VB'), ('다', 'EE'), ('.', 'SY')]
```

#### 7.2 dependency parse

구문분석 결과를 출력한다. 어절 단위로 분석한다.

사용 예시

```python
>>> from nltkor.tag import EspressoTagger
>>> sent = "우리는 아름다운 강산에 살고 있다."

>>> tagger = EspressoTagger(task='dependency')
>>> print(tagger.tag(sent))
[[(1, '우리는', '4', 'NP_SBJ'), (2, '아름다운', '3', 'VP_MOD'), (3, '강산에', '4', 'NP_AJT'), (4, '살고', '5', 'VP'), (5, '있다.', '0', 'VP')]]
```

예제는 '(어절번호, 어절, 지배소 어절번호, 구문관계)'를 의미한다.

#### 7.3 wsd tag

형태 의미 정보를 출력한다.
**사용법 & 결과**

```python
>>> from nltkor.tag import EspressoTagger
>>> sent = "나는 아름다운 강산에 살고있다."

>>> tagger = EspressoTagger(task='wsd')
>>> print(tagger.tag(sent))
['나_*', '는_*', '아름답_*', 'ㄴ_*', '강산_01', '에_*', '살_01', '고_*', '있_*', '다_*', '._*']
```

#### 7.4 ner tag

**사용법 & 결과**

```python
>>> from nltkor.tag import EspressoTagger
>>> sent = "나는 배가 고프다."

>>> tagger = EspressoTagger(task='ner')
>>> print(tagger.tag(sent))
['나_*', '는_*', '배_AM-S', '가_*', '고프_*', '다_*', '._*']
```

#### 7.5 srl tag

**사용법 & 결과**

```python
>>> from nltkor.tag import EspressoTagger
>>> sent = "나는 배가 고프다. 나는 아름다운 강산에 살고있다."

>>> tagger = EspressoTagger(task='srl')
>>> print(tagger.tag(sent))
[{'고프다.': {'ARG0': '나는', 'ARG1': '배가'}}, {'아름다운': {}, '살고있다.': {'ARG0': '나는', 'ARG1': '강산에'}}]
```

### 8. 번역 함수

파파고 번역기를 이용한 한/영 간 번역 기능 함수이다.

- e2k : 영어 ➔ 한국어 변환
- k2e : 한국어 ➔ 영어 변환

```
e2k[k2e](sentence(s))

    :: 파라미터
            sentence(s) : list of str

    :: 결과값
            translation sentence(s) : list of str
```

```python
>>> from nltkor import trans
>>> sent_list = ['넷플릭스를 통해 전 세계 동시 방영돼 큰 인기를 끈 드라마 시리즈 ‘오징어 게임’의 인기가 구글 인기 검색어로도 확인됐다.']
>>> papago = trans.papago()
>>> papago.k2e(sent_list)
['The popularity of the drama series "Squid Game," which was broadcast simultaneously around the world through Netflix and became very popular, has also been confirmed as a popular search term for Google.']
>>> sent_list = ['The popularity of the drama series "Squid Game," which was aired simultaneously around the world through Netflix, has also been confirmed as a popular search term for Google.']
>>> papago.e2k(sent_list)
["넷플릭스를 통해 전 세계 동시 방영된 드라마 시리즈 '오징어 게임'의 인기도 구글의 인기 검색어로 확인됐다."]
```

### 9. 정렬 (Alignment)

두 문장의 정렬 결과를 반환하는 함수이다. 해당 함수는 sting2sting의 코드를 참고하거나 포함하고 있다. (https://github.com/stanfordnlp/string2string)

#### 9.1 Needleman-Wunsch 알고리즘

get_alignment(str1: str|List[str], str2: str|List[str], return_score_matrix: bool = False) -> Tuple(str|List[str], str|List[str], ndarray|None) : 두 문장의 글로벌 정렬 결과를 반환한다.

```python
>>> from nltkor.alignment import NeedlemanWunsch
>>> sent1 = "우리는 생명을 보존해야 한다."
>>> sent2 = "우리가 생명의 존엄을 보존해야지."
>>> result1, result2 = NeedlemanWunsch().get_alignment(sent1, sent2)
>>> print(result1, '\n', result2)
우 | 리 | 는 |   | 생 | 명 | - | - | - | - | 을 |   | 보 | 존 | 해 | 야 |   | 한 | 다 | .
 우 | 리 | 가 |   | 생 | 명 | 의 |   | 존 | 엄 | 을 |   | 보 | 존 | 해 | 야 | - | - | 지 | .
>>> result1, result2, score = NeedlemanWunsch().get_alignment(sent1, sent2, True)
>>> print(result1, '\n', result2, '\n', score)
우 | 리 | 는 |   | 생 | 명 | - | - | - | - | 을 |   | 보 | 존 | 해 | 야 |   | 한 | 다 | .
 우 | 리 | 가 |   | 생 | 명 | 의 |   | 존 | 엄 | 을 |   | 보 | 존 | 해 | 야 | - | - | 지 | .
 [[  0.  -1.  -2.  -3.  -4.  -5.  -6.  -7.  -8.  -9. -10. -11. -12. -13.
  -14. -15. -16. -17. -18.]
 [ -1.   1.   0.  -1.  -2.  -3.  -4.  -5.  -6.  -7.  -8.  -9. -10. -11.
  -12. -13. -14. -15. -16.]
 ...
 [-15. -13. -11. -11.  -9.  -7.  -5.  -5.  -3.  -1.  -1.  -1.   1.   1.
    1.   1.   3.   3.   3.]
 [-16. -14. -12. -12. -10.  -8.  -6.  -6.  -4.  -2.  -2.  -2.   0.   0.
    0.   0.   2.   2.   4.]]
```

#### 9.2 Hirschberg 알고리즘

get_alignment(str1: str|List[str], str2: str|List[str]) -> Tuple(str|List[str], str|List[str]) : 두 문장의 글로벌 정렬 결과를 반환한다.

```python
>>> from nltkor.alignment import Hirschberg
>>> sent1 = "우리는 생명을 보존해야 한다."
>>> sent2 = "우리가 생명의 존엄을 보존해야지."
>>> result1, result2 = Hirschberg().get_alignment(sent1, sent2)
>>> print(result1, '\n', result2)
우 | 리 | 는 |   | 생 | 명 | - | - | - | - | 을 |   | 보 | 존 | 해 | 야 |   | 한 | 다 | .
 우 | 리 | 가 |   | 생 | 명 | 의 |   | 존 | 엄 | 을 |   | 보 | 존 | 해 | 야 | 지 | - | - | .
```

#### 9.3 Smith-Waterman 알고리즘

get_alignment(str1: str | List[str], str2: str | List[str], return_score_matrix: bool = False) -> Tuple[str | List[str], str | List[str]] : 두 문장의 로컬 정렬 결과를 반환한다.

```python
>>> from nltkor.alignment import SmithWaterman
>>> sent1 = "우리는 생명을 보존해야 한다."
>>> sent2 = "우리가 생명의 존엄을 보존해야지."
>>> result1, result2 = SmithWaterman().get_alignment(sent1, sent2)
>>> print(f"{result1}\n{result2}")
을 |   | 보 | 존 | 해 | 야
을 |   | 보 | 존 | 해 | 야
>>> result1, result2, score = SmithWaterman().get_alignment(sent1, sent2, True)
>>> print(f"{result1}\n{result2}\n{score}")
을 |   | 보 | 존 | 해 | 야
을 |   | 보 | 존 | 해 | 야
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 2. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 2. 1. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 3. 2. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 2. 4. 3. 2. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 3. 3. 2. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 2. 2. 4. 3. 2. 1. 2. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 1. 3. 3. 2. 1. 1. 3. 2. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 2. 4. 3. 2. 1. 2. 4. 3. 2. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 3. 3. 2. 1. 1. 3. 5. 4. 3. 2.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 2. 2. 1. 0. 2. 4. 6. 5. 4.]
 [0. 0. 0. 0. 1. 0. 0. 0. 1. 1. 1. 1. 3. 2. 1. 3. 5. 5. 4.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 2. 1. 2. 4. 4. 4.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 3. 3. 3.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 2. 4.]]
```

#### 9.4 DTW

get_alignment_path(sequence1: str | List[str] | int | List[int] | float | List[float] | ndarray, sequence2: str | List[str] | int | List[int] | float | List[float] | ndarray, distance='absolute_difference', p_value: int | None = None) -> List[Tuple[int, int]] : DTW 알고리즘을 사용하여 두 시퀀스의 정렬 인덱스를 반환한다.

- absolute_difference: |a-b| (기본)
- square_difference: (a-b)\*\*2

```python
>>> from nltkor.alignment import DTW
>>> sent1 = "우리는 생명을 보존해야 한다."
>>> sent2 = "우리가 생명의 존엄을 보존해야지."
>>> result = DTW().get_alignment_path(sent1, sent2)
>>> print(result)
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 8), (10, 8), (11, 9), (11, 10), (12, 11), (13, 12), (13, 13), (13, 14), (13, 15), (14, 16), (15, 17)]
>>> result = DTW().get_alignment_path(sent1, sent2, distance='square_difference')
>>> print(result)
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 8), (10, 8), (11, 9), (11, 10), (12, 11), (13, 12), (13, 13), (13, 14), (13, 15), (14, 16), (15, 17)]
```

#### 9.5 Longest Common Subsequence

compute(str1: str | List[str], str2: str | List[str], returnCandidates: bool = False) -> Tuple[float, List[str] | List[List[str]]] : 두 문자열의 가장 긴 공통 서브시퀀스를 계산한다.

```python
>>> from nltkor.alignment import LongestCommonSubsequence
>>> sent1 = "우리는 생명을 보존해야 한다."
>>> sent2 = "우리가 생명의 존엄을 보존해야지."
>>> result = LongestCommonSubsequence().compute(sent1, sent2)
>>> print(result)
(12.0, None)
>>> result = LongestCommonSubsequence().compute(sent1, sent2, True)
>>> print(result)
(12.0, ['우리 생명을 보존해야.'])
```

#### 9.6 Longest Common Substring

compute(str1: str | List[str], str2: str | List[str], returnCandidates: bool = False) -> Tuple[float, List[str] | List[List[str]]] : 두 문자열의 가장 긴 공통 서브스트링을 계산한다.

```python
>>> from nltkor.alignment import LongestCommonSubstring
>>> sent1 = "우리는 생명을 보존해야 한다."
>>> sent2 = "우리가 생명의 존엄을 보존해야지."
>>> result = LongestCommonSubstring().compute(sent1, sent2)
>>> print(result)
(6, None)
>>> result = LongestCommonSubstring().compute(sent1, sent2, True)
>>> print(result)
(6, ['을 보존해야'])
```

### 10. 거리(Distance)

해당 함수는 sting2sting의 코드를 참고하거나 포함하고 있다. (https://github.com/stanfordnlp/string2string)

#### 10.1 Levenshtein Edit Distance

- compute(str1: str | List[str], str2: str | List[str], method: str = 'dynamic-programming')→ float : 두 문자열의 Levenshtein 거리를 계산한다.
  - method : 'dynamic-programming' | 'recursive' | 'recursive-memoization'

```python
>>> from nltkor.distance import LevenshteinEditDistance
>>> sent1 = "우리는 생명을 보존해야 한다."
>>> sent2 = "우리가 생명의 존엄을 보존해야지."
>>> result = LevenshteinEditDistance().compute(sent1, sent2)
>>> print(result)
8.0
>>> result = LevenshteinEditDistance(match_weight=0.1, insert_weight=0.9, delete_weight=0.8, substitute_weight=0.9).compute(sent1, sent2)
>>> print(result)
8.199999999999998
```

#### 10.2 Hamming Distance

- compute(str1: str | List[str], str2: str | List[str])→ float : 두 문자열의 Hamming 거리를 계산한다.
- 두 문자열의 길이가 같아야 한다.

```python
>>> from nltkor.distance import HammingDistance
>>> sent1 = "우리는 생명을 보존해야 한다."
>>> sent2 = "우리가 생명의 존엄을 보존해."
>>> result = HammingDistance().compute(sent1, sent2)
>>> print(result)
9.0
>>> result = HammingDistance(match_weight=0.1, substitute_weight=0.9).compute(sent1, sent2)
>>> print(result)
8.8
```

#### 10.3 Damereau-Levenshtein Distance

compute(str1: str | List[str], str2: str | List[str]) -> float : 두 문자열의 Damereau-Levenshtein 거리를 계산한다.

```python
>>> from nltkor.distance import DamerauLevenshteinDistance
>>> sent1 = "우리는 생명을 보존해야 한다."
>>> sent2 = "우리가 생명의 존엄을 보존해야지."
>>> result = DamerauLevenshteinDistance().compute(sent1, sent2)
>>> print(result)
8.0
>>> result = DamerauLevenshteinDistance(match_weight=0.1, insert_weight=0.9, delete_weight=0.8, substitute_weight=0.9, adjacent_transpose_weight=0.8).compute(sent1, sent2)
>>> print(result)
8.199999999999998
```

#### 10.4 Wasserstein Distance

- compute_kullback(p: np.ndarray | torch.Tensor, q: np.ndarray | torch.Tensor) -> float : 두 Tensor간의 Kullback-Leibler 거리를 계산한다.
- compute_wasserstein(p: np.ndarray | torch.Tensor, q: np.ndarray | torch.Tensor) -> float : 두 Tensor간의 Wasserstein 거리를 계산한다.
- compute_jesson_shannon(p: np.ndarray | torch.Tensor, q: np.ndarray | torch.Tensor) -> float : 두 Tensor간의 jesson_shannon 거리를 계산한다.

```python
>>> from nltkor.distance import WassersteinDistance
>>> import torch
>>> import numpy as np
>>> P =  np.array([0.6, 0.1, 0.1, 0.1, 0.1])
>>> Q1 = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
>>> Q2 = np.array([0.1, 0.1, 0.1, 0.1, 0.6])
>>> P = torch.from_numpy(P)
>>> Q1 = torch.from_numpy(Q1)
>>> Q2 = torch.from_numpy(Q2)
>>> kl_p_q1 = WassersteinDistance().compute_kullback(P, Q1)
>>> kl_p_q2 = WassersteinDistance().compute_kullback(P, Q2)
>>> wass_p_q1 = WassersteinDistance().compute_wasserstein(P, Q1)
>>> wass_p_q2 = WassersteinDistance().compute_wasserstein(P, Q2)
>>> print("\nKullback-Leibler distances: ")

Kullback-Leibler distances:
>>> print("P to Q1 : %0.4f " % kl_p_q1)
P to Q1 : 1.7918
>>> print("P to Q2 : %0.4f " % kl_p_q2)
P to Q2 : 1.7918
>>> print("\nWasserstein distances: ")

Wasserstein distances:
>>> print("P to Q1 : %0.4f " % wass_p_q1)
P to Q1 : 1.0000
>>> print("P to Q2 : %0.4f " % wass_p_q2)
P to Q2 : 2.0000
>>> jesson_p_q1 = WassersteinDistance().compute_jesson_shannon(P, Q1)
>>> jesson_p_q2 = WassersteinDistance().compute_jesson_shannon(P, Q2)
>>> print("\nJesson-Shannon distances: ")

Jesson-Shannon distances:
>>> print("P to Q1 : %0.4f " % jesson_p_q1)
P to Q1 : 0.1981
>>> print("P to Q2 : %0.4f " % jesson_p_q2)
P to Q2 : 0.1981
```

### 11. 유사도 (Similarity)

해당 함수는 sting2sting의 코드를 참고하거나 포함하고 있다. (https://github.com/stanfordnlp/string2string)

#### 11.1 코사인 유사도 (Cosine Similarity)

- compute(x1: Tensor | ndarray, x2: Tensor | ndarray, dim: int = 0, eps: float = 1e-08) -> Tensor | ndarray : 두 벡터의 코사인 유사도를 계산한다.

```python
>>> from nltkor.similarity import CosineSimilarity
>>> import numpy as np
>>> x1 = np.array([1, 2, 3, 4, 5])
>>> x2 = np.array([3, 7, 8, 3, 1])
>>> result = CosineSimilarity().compute(x1, x2)
>>> print(result)
0.6807061638788793
```

#### 11.2 LCSubstring Similarity

- compute(str1: str | List[str], str2: str | List[str], denominator: str = 'max') -> float : 두 문자열의 LCSubstring 유사도를 계산한다.
  - denominator : (max | sum)

```python
>>> from nltkor.similarity import LCSubstringSimilarity
>>> sent1 = "우리는 생명을 보존해야 한다."
>>> sent2 = "우리가 생명의 존엄을 보존해야지."
>>> result = LCSubstringSimilarity().compute(sent1, sent2)
>>> print(result)
0.3333333333333333
>>> result = LCSubstringSimilarity().compute(sent1, sent2, denominator='sum')
>>> print(result)
0.35294117647058826
```

#### 11.3 LCSubsequence Similarity

- compute(str1: str | List[str], str2: str | List[str], denominator: str = 'max') -> float : 두 문자열의 LCSubsequence 유사도를 계산한다.
  - denominator : (max | sum)

```python
>>> from nltkor.similarity import LCSubsequenceSimilarity
>>> sent1 = "우리는 생명을 보존해야 한다."
>>> sent2 = "우리가 생명의 존엄을 보존해야지."
>>> result = LCSubsequenceSimilarity().compute(sent1, sent2)
>>> print(result)
0.6666666666666666
>>> result = LCSubsequenceSimilarity().compute(sent1, sent2, denominator='sum')
>>> print(result)
0.7058823529411765
```

#### 11.4 Jaro Similarity

- compute(str1: str | List[str], str2: str | List[str])→ float : 두 문자열의 Jaro 유사도를 반환한다.

```python
>>> from nltkor.similarity import JaroSimilarity
>>> sent1 = "우리는 생명을 보존해야 한다."
>>> sent2 = "우리가 생명의 존엄을 보존해야지."
>>> result = JaroSimilarity().compute(sent1, sent2)
>>> print(result)
0.7679843304843305
```

### 12. 검색 (Search)

해당 함수는 sting2sting의 코드를 참고하거나 포함하고 있다. (https://github.com/stanfordnlp/string2string)

#### 12.1 Navie Search

- search(pattern: str, text: str) -> int : 텍스트에서 패턴을 검색한다.

```python
>>> from nltkor.search import NaiveSearch
>>> pattern = '생명'
>>> text = "우리는 생명을 보존해야 한다."
>>> result = NaiveSearch().search(pattern, text)
>>> print(result)
4
```

#### 12.2 Rabin-Karp 검색

- search(pattern: str, text: str) -> int : 텍스트에서 패턴을 검색한다.

```python
>>> from nltkor.search import RabinKarpSearch
>>> pattern = '생명'
>>> text = "우리는 생명을 보존해야 한다."
>>> result = RabinKarpSearch().search(pattern, text)
>>> print(result)
4
```

#### 12.3 KMP 검색

- search(pattern: str, text: str) -> int : 텍스트에서 패턴을 검색한다.

```python
>>> from nltkor.search import KMPSearch
>>> pattern = '생명'
>>> text = "우리는 생명을 보존해야 한다."
>>> result = KMPSearch().search(pattern, text)
>>> print(result)
4
```

#### 12.4 Boyer-Moore 검색

- search(pattern: str, text: str) -> int : 텍스트에서 패턴을 검색한다.

```python
>>> from nltkor.search import BoyerMooreSearch
>>> pattern = '생명'
>>> text = "우리는 생명을 보존해야 한다."
>>> result = BoyerMooreSearch().search(pattern, text)
>>> print(result)
4
```

#### 12.5 Faiss-Semantic 검색

- **init**(model_name_or_path: str = 'facebook/bart-large', tokenizer_name_or_path: str = 'facebook/bart-large', device: str = 'cpu')→ None : FaissSearh를 초기화 합니다.
- add_faiss_index(column_name: str = 'embeddings', metric_type: int | None = None, batch_size: int = 8, \*\*kwargs)→ None : FAISS index를 dataset에 추가합니다.
- get_embeddings(text: str | List[str], embedding_type: str = 'last_hidden_state', batch_size: int = 8, num_workers: int = 4)→ Tensor : 텍스트를 임베딩합니다.
- get_last_hidden_state(embeddings: Tensor)→ Tensor : 임베딩된 텍스트의 last hidden state를 반환합니다.
- get_mean_pooling(embeddings: Tensor)→ Tensor : 입력 임베딩의 mean pooling을 반환합니다.
- initialize_corpus(corpus: Dict[str, List[str]] | DataFrame | Dataset, section: str = 'text', index_column_name: str = 'embeddings', embedding_type: str = 'last_hidden_state', batch_size: int | None = None, num_workers: int | None = None, save_path: str | None = None)→ Dataset : 데이터셋을 초기화합니다.
- load_dataset_from_json(json_path: str)→ Dataset : json 파일에서 데이터셋을 로드합니다.
- load_faiss_index(index_name: str, file_path: str, device: str = 'cpu')→ None : FAISS index를 로드합니다.
- save_faiss_index(index_name: str, file_path: str)→ None : 특정한 파일 경로로 FAISS index를 저장합니다.
- search(query: str, k: int = 1, index_column_name: str = 'embeddings')→ DataFrame : 데이터셋에서 쿼리를 검색합니다.

```python
>>> from nltkor.search import FaissSearch
>>> faiss = FaissSearch(model_name_or_path = 'facebook/bart-large')
>>> corpus = {
        'text': [
                "오늘은 날씨가 매우 덥습니다.",
                "저는 음악을 듣는 것을 좋아합니다.",
                "한국 음식 중에서 떡볶이가 제일 맛있습니다.",
                "도서관에서 책을 읽는 건 좋은 취미입니다.",
                "내일은 친구와 영화를 보러 갈 거예요.",
                "여름 휴가 때 해변에 가서 수영하고 싶어요.",
                "한국의 문화는 다양하고 흥미로워요.",
                "피아노 연주는 나를 편안하게 해줍니다.",
                "공원에서 산책하면 스트레스가 풀립니다.",
                "요즘 드라마를 많이 시청하고 있어요.",
                "커피가 일상에서 필수입니다.",
                "새로운 언어를 배우는 것은 어려운 일이에요.",
                "가을에 단풍 구경을 가고 싶어요.",
                "요리를 만들면 집안이 좋아보입니다.",
                "휴대폰 없이 하루를 보내는 것이 쉽지 않아요.",
                "스포츠를 하면 건강에 좋습니다.",
                "고양이와 개 중에 어떤 동물을 좋아하세요?"
                "천천히 걸어가면서 풍경을 감상하는 것이 좋아요.",
                "일주일에 한 번은 가족과 모임을 가요.",
                "공부할 때 집중력을 높이는 방법이 있을까요?",
                "봄에 꽃들이 피어날 때가 기대되요.",
                "여행 가방을 챙기고 싶어서 설레여요.",
                "사진 찍는 걸 좋아하는데, 카메라가 필요해요.",
                "다음 주에 시험이 있어서 공부해야 해요.",
                "운동을 하면 몸이 가벼워집니다.",
                "좋은 책을 읽으면 마음이 풍요로워져요.",
                "새로운 음악을 발견하면 기분이 좋아져요.",
                "미술 전시회에 가면 예술을 감상할 수 있어요.",
                "친구들과 함께 시간을 보내는 건 즐거워요.",
                "자전거 타면 바람을 맞으면서 즐거워집니다."
        ],
    }
>>> faiss.initialize_corpus(corpus=corpus, section='text', embedding_type='mean_pooling')
>>> query = "오늘은 날시가 매우 춥다."
>>> top_k = 5
>>> result = faiss.search(query, top_k)
>>> print(result)
                       text                                         embeddings      score
0          오늘은 날씨가 매우 덥습니다.  [-0.2576247453689575, 0.47791656851768494, -1....  14.051050
1  한국 음식 중에서 떡볶이가 제일 맛있습니다.  [-0.2623925805091858, 0.46345704793930054, -1....  28.752083
2      요즘 드라마를 많이 시청하고 있어요.  [-0.2683958113193512, 0.6801461577415466, -1.1...  29.339230
3    다음 주에 시험이 있어서 공부해야 해요.  [-0.20012563467025757, 0.5758355855941772, -1....  31.358824
4     피아노 연주는 나를 편안하게 해줍니다.  [-0.24231986701488495, 0.6492734551429749, -1....  34.069862
```

- faiss 검색을 매번 initialize 하지 않고, 미리 initialize 해놓은 후 검색을 수행할 수 있습니다.

**사용법 & 결과**

```python
>>> from nltkor.search import FaissSearch

# if you use model and tokenizer in local
# faiss = FaissSearch(model_name_or_path = '~/test_model/trained_model/', tokenizer_name_or_path = '~/test_model/trained_model/')

>>> faiss = FaissSearch(model_name_or_path = 'facebook/bart-large')
>>> corpus = {
        'text': [
                "오늘은 날씨가 매우 덥습니다.",
                "저는 음악을 듣는 것을 좋아합니다.",
                "한국 음식 중에서 떡볶이가 제일 맛있습니다.",
                "도서관에서 책을 읽는 건 좋은 취미입니다.",
                "내일은 친구와 영화를 보러 갈 거예요.",
                "여름 휴가 때 해변에 가서 수영하고 싶어요.",
                "한국의 문화는 다양하고 흥미로워요.",
                "피아노 연주는 나를 편안하게 해줍니다.",
                "공원에서 산책하면 스트레스가 풀립니다.",
                "요즘 드라마를 많이 시청하고 있어요.",
                "커피가 일상에서 필수입니다.",
                "새로운 언어를 배우는 것은 어려운 일이에요.",
                "가을에 단풍 구경을 가고 싶어요.",
                "요리를 만들면 집안이 좋아보입니다.",
                "휴대폰 없이 하루를 보내는 것이 쉽지 않아요.",
                "스포츠를 하면 건강에 좋습니다.",
                "고양이와 개 중에 어떤 동물을 좋아하세요?"
                "천천히 걸어가면서 풍경을 감상하는 것이 좋아요.",
                "일주일에 한 번은 가족과 모임을 가요.",
                "공부할 때 집중력을 높이는 방법이 있을까요?",
                "봄에 꽃들이 피어날 때가 기대되요.",
                "여행 가방을 챙기고 싶어서 설레여요.",
                "사진 찍는 걸 좋아하는데, 카메라가 필요해요.",
                "다음 주에 시험이 있어서 공부해야 해요.",
                "운동을 하면 몸이 가벼워집니다.",
                "좋은 책을 읽으면 마음이 풍요로워져요.",
                "새로운 음악을 발견하면 기분이 좋아져요.",
                "미술 전시회에 가면 예술을 감상할 수 있어요.",
                "친구들과 함께 시간을 보내는 건 즐거워요.",
                "자전거 타면 바람을 맞으면서 즐거워집니다."
        ],
    }
>>> faiss.initialize_corpus(corpus=corpus, section='text', embedding_type='mean_pooling', save_path='./test.json')
```

- `initialize_corpus()` 메소드 실행시 `save_path`를 지정하면, 해당 경로에 임베딩된 Dataset이 json형식으로 저장됩니다.

```python
>>> from nltkor.search import FaissSearch

>>> faiss = FaissSearch(model_name_or_path = 'facebook/bart-large')
>>> faiss.load_dataset_from_json('./test.json')
>>> faiss.embedding_type = 'mean_pooling' # initalize_corpus() 메소드 실행시 지정한 embedding_type과 동일하게 지정해야 합니다.
>>> faiss,add_faiss_index(colum_name = 'embeddings')
>>> query = '오늘은 날씨가 매우 춥다.'
>>> top_k = 5
>>> result = faiss.search(query=query, top_k=top_k)
>>> print(result)
Downloading data files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 6052.39it/s]
Extracting data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 333.68it/s]
Generating train split: 29 examples [00:00, 2969.89 examples/s]
Adding FAISS index...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 3575.71it/s]
                       text                                         embeddings      score
0          오늘은 날씨가 매우 덥습니다.  [-0.2576247454, 0.4779165685, -1.3404077291, 0...  14.051050
1  한국 음식 중에서 떡볶이가 제일 맛있습니다.  [-0.2623925805, 0.4634570479, -1.1966289282, 0...  28.752083
2      요즘 드라마를 많이 시청하고 있어요.  [-0.2683958113, 0.6801461577, -1.1375769377, 0...  29.339230
3    다음 주에 시험이 있어서 공부해야 해요.  [-0.2001256347, 0.5758355856, -1.0528291464, 0...  31.358824
4     피아노 연주는 나를 편안하게 해줍니다.  [-0.242319867, 0.6492734551, -1.4172941446, 0....  34.069862
```

#### 12.6 TRIE 검색
- 텍스트 파일에 word가 포함되어 있는지 판단한다.

```python
root = {}
dict_file = '텍스트 파일 경로'
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
values, value_data = sc.search_dict(word, True)
print(values, value_data)
word = '2시 뉴스외전'
values, value_data = sc.search_dict(word, True)
print(values, value_data)

word = 'gbc'
values, value_data = sc.search_dict(word, True)
print(values, value_data)
```
**결과**
```
['고용 노동부'] ['NN']
['2시뉴스외전'] ['NN']
['2시 뉴스외전'] ['NN']
['bc'] ['ND']
```

### 13. 세종전자사전 (ssem)

우선 해당 기능을 사용하기 전에 인자 포맷에 대해 설명한다. 인자는 **entrys, entry, sense** 함수에서 사용한다. 인자 포맷을 설명하기 위해 예제는 체언의 '눈'과 용언의 '감다'를 이용하였다. 

<center><b>인자 포맷 : ' 단어(.형태소)((.entry번호).sense번호) '</b></center>

```h
entrys : '단어' 명시
ex) entrys('눈'), entrys('감다')

entry : '단어 & 형태소 & entry 번호' 명시
ex) entry('눈.nng_s.1'), entry('감다.vv.1')

sense : '단어 & 형태소 & entry 번호 & sense 번호' 명시
ex) sense('눈.nng_s.1.1'), sense('감다.vv.1.01')
```

|   분 류    | 설 명                                                                                                                                                                           |
| :--------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|    단어    | 사용자 검색 단어                                                                                                                                                                |
|   형태소   | 검색 단어의 형태소 <br>**체언 : nng_s(단일어 명사 : '강'), nng_c(복합어 명사 : '강물') / 용언 : vv,va**                                                                         |
| entry 번호 | 해당 단어와 형태소의 entry 번호 <br>(entry에 직접 접근을 위해 필요)<br>**체언 , 용언 : 1,2,3,...,n**                                                                            |
| sense 번호 | 해당 단어와 형태소, entry이하 sense 번호<br>(sense에 직접 접근을 위하여 필요)<br>**체언 : 1,2,3, ...,n (한 자리 숫자로 적용) / 용언 : 01, 02, 03, ...,n (두 자리 숫자로 적용)** |

##### 13.1 객체 확인 방법

filecheck() 함수를 통하여 해당 단어가 세종전자사전의 존재 여부를 확인할 수 있다. 세종전자사전 내 단어가 포함되어 있으면 해당 파일의 위치의 리스트를 반환하고, 포함되지 않는다면 빈 리스트를 반환한다.

```python
>>>from nltkor.sejong import ssem

# 세종전자사전 내 단어 포함여부 확인
>>>check=ssem.filecheck('눈')
['./01. 체언_상세/눈.xml']

>>>check=ssem.filecheck('한라봉')
[]
```

##### 13.2 entry 접근법

아래는 entry에 직•간접적으로 접근하기 위한 방법이다. entrys()를 통해 접근 하거나 entry()를 통해 바로 접근이 가능하다.

```python
>>>from nltkor.sejong import ssem

# 해당 단어가 가지는 entrys 리스트 반환
>>>entrys=ssem.entrys('눈')
[Entry('눈.nng_s.1'), Entry('눈.nng_s.2'), Entry('눈.nng_s.3'), Entry('눈.nng_s.4'), Entry('눈.nng_s.5')]

# entry 객체에 직접 접근
>>>entry=ssem.entry('눈.nng_s.1')
Entry('눈.nng_s.1')

# entry 객체에 간접 접근
>>>entrys=ssem.entrys('눈')[0]
Entry('눈.nng_s.1')
```

##### 13.3 sense 접근법

아래는 sense에 직•간접적 접근하기 위한 방법이다. senses()를 통해 접근하거나, sense()를 통해 바로 접근이 가능하다.

```python
>>>from nltkor.sejong import ssem

entrys=ssem.entrys('눈')
entry=entrys[0]

# 해당 entry가 가지는 sense리스트 반환
>>>senses=entry.senses()
[Sense('눈.nng_s.1.1'),Sense('눈.nng_s.1.2'),Sense('눈.nng_s.1.3'),..., Sense('눈.nng_s.1.7')]

# sense 객체에 간접 접근
>>> sense=senses[0]
Sense('눈.nng_s.1.1')

# sense 객체에 직접 접근
>>>sense=ssem.sense('눈.nng_s.1.1') #체언
Sense('눈.nng_s.1.1')

>>>sense=ssem.sense('있다.va.1.01')  #용언(형용사)
Sense('있다.va.1.01')

>>>sense=ssem.sense('먹다.vv.1.01')  #용언(동사)
Sense('먹다.vv.1.01')
```

##### 13.4 entry 함수 사용법 & 결과

entry에 직•간접적 접근을 통해 사용할 수 있는 3가지 함수(파생어, 복합어, 파생어)의 결과를 리스트로 반환한다.

```python
>>> from nltkor.sejong import ssem

# entry 객체에 직접 접근
>>> entry=ssem.entry('눈.nng_s.2')

# 해당 entry 객체의 숙어 리스트 반환
>>>entry.idm()
['눈이 오나 비가 오나']

# 해당 entry 객체의 복합어 리스트 반환
>>>entry.comp()
['눈사람', '눈길', '눈송이', '눈싸움', '눈더미', '함박눈', '싸락눈']

# 해당 entry 객체의 파생어 리스트 반환
>>>entry.der()
['눈발']
```

##### 13.5 sense 함수 사용법 & 결과

sense의 메소드를 사용하기 위해서는 sense에 직접 접근하는 방법과 그 외 entrys, entry에서 접근하는 방법이 있다.

아래는 직접 접근하여 sense의 메소드를 알아보는 예이다. 해당 결과가 없다면 빈 리스트를 반환한다. 단어의 sem, 동의어, 반의어, 동위어, 상위어, 하위어, 전체어, 부분어, 관련어, 영어, 예시, 형용사 결합, 명사 결합, 동사 결합, 선택제약, 경로, 유사도 반환 함수 17개를 포함한다. wup_similarity()는 두 단어의 유사도 비교를 위하여 target이 되는 단어가 인자로 들어오게 된다.

```python
from nltkor.sejong import ssem

# sense 접근 방법 1 (직접 접근)
>>>sense=ssem.sense('눈.nng_s.1.1')
Sense('눈.nng_s.1.1')

# sense 접근 방법 2 (간접 접근 1)
>>>entrys=ssem.entrys('눈')
[Entry('눈.nng_s.1'), Entry('눈.nng_s.2'), Entry('눈.nng_s.3'), Entry('눈.nng_s.4'), Entry('눈.nng_s.5')]
>>>sense = entrys[0].senses()[0]
Sense('눈.nng_s.1.1')

# sense 접근 방법 3 (간접 접근 2)
>>>sense=ssem.entry('눈.nng_s.1').senses()[0]
Sense('눈.nng_s.1.1')

# 해당 sense의 sem 확인
>>>sense.sem()
['신체부위']

# 해당 sense의 동의어 리스트 반환
>>>sense.syn()
[]

# 해당 sense의 반의어 리스트 반환
>>>sense.ant()
[]

# 해당 sense의 동위어 리스트 반환
>>>sense.coord()
['위', '입', '코', '이마', '머리', '볼']

# 해당 sense의 상위어 리스트 반환
>>>sense.hyper()
['신체부위']

# 해당 sense의 하위어 리스트 반환
>>>sense.hypo()
[]

# 해당 sense의 전체어 리스트 반환
>>>sense.holo()
['얼굴']

# 해당 sense의 부분어 리스트 반환
>>>sense.mero()
['눈동자', '눈시울', '눈망울', '눈꺼풀']

# 해당 sense의 관련어 리스트 반환
>>>sense.rel()
[]

# 해당 sense의 영어 리스트 반환
>>>sense.trans()
['eye']

# 해당 sense의 예시 리스트 반환
>>>sense.example()
['그는 눈을 크게 뜨고 나를 바라보았다.', '그 아이는 엄마를 닮아 눈이 아주 컸다.']

# 해당 sense의 형용사결합 리스트 반환
>>>sense.comb_aj()
['눈이 아름답다', '눈이 초롱초롱하다', '눈이 아프다', '눈이 맑다', '눈이 크다', '눈이 작다', '눈이 가늘다']

# 해당 sense의 명사결합 리스트 반환
>>>sense.comb_n()
[]

# 해당 sense의 동사결합 리스트 반환
>>>sense.comb_v()
['눈을 뜨다', '눈을 감다', '눈을 흘기다', '눈이 감기다', '눈이 휘둥그레지다']

# 해당 sense의 선택제약 리스트 반환
>>>sense.sel_rst()
[]

# 해당 sense의 상위 경로 리스트 반환
>>>sense.sem_path()
['1_구체물', '1.4_관계구체물', '1.4.1_부분', '1.4.1.2_신체부위']

# semclass 기반의 두 단어 유사도 계산 반환하며, 0~1의 값을 가진다.
# 'target': 유사도를 계산하기 위한 단어
>>>target=ssem.sense('발.nng_s.1.1')
>>>sense.wup_similarity(target)
0.8
```

### 14. etc

#### 14.1

- parse_morph : 예제 확인

- TBA

**사용법 & 결과**

```python
>>> from nltkor import etc

'''

parse_morphs(target)
    * args
            target : 'word1/pos + word2/pos+...'

    * return : [('word1',pos), ('word2',pos)...]
'''

>>> org = '하늘 / NNG + 을 / JKO + 날 / VV + 는 / ETD + 자동차 / NNG'
>>> etc.parse_morph(org)
[('하늘', 'NNG'), ('을', 'JKO'), ('날', 'VV'), ('는', 'ETD'), ('자동차', 'NNG')]


>>> org = '아버지/NNG + 가/JKS + 방/NNG + 에/JKB + 들어가/VV + 신다/EP+EC'
>>> etc.parse_morph(org)
[('아버지', 'NNG'), ('가', 'JKS'), ('방', 'NNG'), ('에', 'JKB'), ('들어가', 'VV'), ('신다', 'EP')]
```
