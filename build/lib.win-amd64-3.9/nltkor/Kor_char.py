import unicodedata

CHOSEONG_IDX_CODEMAP = [1, 2, 0, 3, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 7, 8, 9, 0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
JONGSEONG_IDX_CODEMAP= [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 18, 19, 20, 21, 22, 0, 23, 24, 25, 26, 27]
getCJamoIdxChoseong  = lambda x: ((x > min(CHOSEONG_IDX_CODEMAP) and x <= max(CHOSEONG_IDX_CODEMAP)) and CHOSEONG_IDX_CODEMAP.index(x)) or 0
getCJamoIdxJongseong = lambda x: ((x > min(JONGSEONG_IDX_CODEMAP) and x <= max(JONGSEONG_IDX_CODEMAP)) and JONGSEONG_IDX_CODEMAP.index(x)) or 0



def error():

	try:
		raise Exception("function expect a character, check the value")
	except Exception as e:
		print(e)
		return

#한글 문자 판별 함수. 구: kor_check()
def is_kor_char(character, encoding = None):

	if len(character)>1 or len(character)<=0:
		error()
		return

	ch = ord(character)

	return (	( ch >= 0xac00 and ch <= 0xd7a3)	or	# Hangul Syllables
				( ch >= 0x1100 and ch <= 0x11ff)	or	# Hangul Jamo
				( ch >= 0x3131 and ch <= 0x318e)	or	# Hangul Compatibility Jamo
				( ch >= 0xffa1 and ch <= 0xffdc)	)	# Hangul Halfwidth


# 초/중/종성 분할 함수. kor_split()
def split_syllable(character, encoding = None):

	if len(character)>1 or len(character)<=0:
		error()
		return

	char = character
	returnChr = lambda x: (x and chr(x)) or str()
	returnCJJ = lambda x, y, z: tuple(map(returnChr, (x, y, z)))

	ch = ord(char)


	#초성 : comp
	if (ch >= 0x3131 and ch <= 0x314e) or (ch >= 0x3165 and ch <= 0x3186):
		return returnCJJ(ch, 0, 0)

	#중성 : comp
	if (ch >= 0x314f and ch <= 0x3163) or (ch >= 0x3187 and ch <= 0x318e):
		return returnCJJ(0, ch, 0)

	# Hangul Syllables : 가 - 힣
	if (ch >= 0xac00 and ch <= 0xd7a3):
		idx_cho = int((ch - 0xac00) / 0x024c) # idx_cho = int (ch-44032)/588
		idx_jung= int(((ch - 0xac00) % 0x024c) / 0x001c)#idx_jung = int ((ch-44032)%588)/28
		idx_jong= int((ch - 0xac00) % 0x001c)# idx_jong = int ((ch-44032) % 28)
		return returnCJJ(getCJamoIdxChoseong(idx_cho+1)+0x3131, idx_jung+0x314f, (idx_jong and getCJamoIdxJongseong(idx_jong)+0x3131) or 0)

	# None
	return returnCJJ(ch, 0, 0)



#초/중/종성 결합 함수. kor_join()
def join_syllable(choseong, jungseong, jongseong, encoding = None):


	if len(choseong)|len(choseong)|len(choseong)>1 :
		error()
		return
	elif len(choseong)|len(choseong)|len(choseong)<=0 :
		error()
		return

	returnChr = lambda x: (x and chr(x)) or str()
	returnChar = lambda x: returnChr(x)

	if not jungseong:
		if not choseong:
			return returnChar(0)
		return choseong
	else:
		if not choseong:
			return jungseong

	idx_cho  = CHOSEONG_IDX_CODEMAP[ord(choseong)-0x3131]-1
	idx_jung = ord(jungseong)-0x314f
	idx_jong = (jongseong and JONGSEONG_IDX_CODEMAP[ord(jongseong)-0x3131]) or 0

	return returnChar(0xac00+((idx_cho*21)+idx_jung)*28+idx_jong)



#한글 비교 함수
def kor_cmp(s1, s2, encoding = None):

	if len(s1)|len(s2)>1 or len(s1)|len(s2)<=0 :
		error()
		return


	if type(s1) == str:
		s1 = str().join(map(lambda x: str().join(map(lambda y: y or " ", split_syllable(x))), s1))
	if type(s2) == str:
		s2 = str().join(map(lambda x: str().join(map(lambda y: y or " ", split_syllable(x))), s2))

	return (s1>s2)-(s1<s2)



#한글 음절 판별 함수
def is_kor_syllable(character, encoding = None):

	if len(character)>1 or len(character)<=0:
		error()
		return

	return "HANGUL SYLLABLE" in unicodedata.name(character)



#한자 문자 판별 함수
def is_hanja(character, encoding = None):

	if len(character)>1 or len(character)<=0:
		error()
		return

	return "CJK" in unicodedata.name(character)



#숫자 판별 함수
def is_number(character, encoding = None):

	if len(character)>1 or len(character)<=0:
		error()
		return

	return "DIGIT" in unicodedata.name(character)


#영어 알파벳 문자 판별 함수
def is_eng_char(character, encoding = None):

	if len(character)>1 or len(character)<=0:
		error()
		return

	return "LATIN" in unicodedata.name(character)


#기호 판별 함수
def is_symbol(character, encoding = None):

	if len(character)>1 or len(character)<=0:
		error()
		return

	return unicodedata.category(character)[0] == "S"


#구두점 판별 함수
def is_punctuation(character, encoding = None):

	if len(character)>1 or len(character)<=0:
		error()
		return

	return unicodedata.category(character)[0] == "P"


#영어 알파벳 연결 문자 판별 함수
def is_engConnection(character, encoding = None):

	if len(character)>1 or len(character)<=0:
		error()
		return

	return character in (".", "-", "_", "|")


# 숫자 연결 문자 판별 함수
def is_numConnection(character, encoding = None):

	if len(character)>1 or len(character)<=0:
		error()
		return

	return character in (".", ",")
