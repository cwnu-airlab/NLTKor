import requests
import json
from datetime import datetime
import hmac
import base64
import uuid
import time
import re
from bs4 import BeautifulSoup as bs


class papago: 

	def __init__(self):
		
		
		response=requests.get('https://papago.naver.com')
		html=bs(response.text,'html.parser')
		pattern1=r'/vendors~main.*chunk.js'

		for tmp in html.find_all('script'):
			tmp=str(tmp)
			m=re.search(pattern1,tmp)
			if m is not None:
				a=m.group()

		js_url='https://papago.naver.com'+str(a)
		rest=requests.get(js_url)
		org=rest.text
		pattern2=r'AUTH_KEY:[\s]*"[\w.]+"'
		self.match=str(re.findall(pattern2,org)).split('"')[1]

	# headers 보안키 생성
	def hmac_md5(self,key, s):
		return base64.b64encode(hmac.new(key.encode('utf-8'), s.encode('utf-8'), 'MD5').digest()).decode()


	def translate(self,data,source,target):

		url = 'https://papago.naver.com/apis/n2mt/translate'
		AUTH_KEY = self.match

		dt = datetime.now()
		timestamp = str(round(dt.timestamp()*1000))

		# 고정 값을 사용할 시 서버로 부터 차단을 방지
		deviceId = str(uuid.uuid4())

		headers = {
				'authorization': 'PPG ' + deviceId + ':' + self.hmac_md5(AUTH_KEY, deviceId + '\n' + url + '\n' + timestamp),
				'timestamp': timestamp
				}

		form_data = {
				'deviceId': deviceId,
				'locale': 'ko',
				'dict': 'true',
				'dictDisplay': 30,
				'honorific': 'false',
				'instant': 'false',
				'paging': 'false',
				'source': source,
				'target': target,
				'text': data
				}

		res_data = requests.post(url, data=form_data, headers=headers)

		#papago 번역 결과물 전체 확인
		#print("\n\n\n",res_data.json())

		return res_data.json()['translatedText']


	def e2k(self,sent_list):

		patient = 0
		return_list=[]

		for line in sent_list:
			line = line.strip()
			try:
				text = self.translate(line,'en','ko') ## translatin
			except (KeyError,requests.exceptions.ConnectionError) as e:
				if patient > 5:
					ofp.close()
					exit() ## Error가 5번 이상 누적되면 종료
				patient += 1
				time.sleep(30) ## 에러 발생 시 1시간 대기
				continue

			return_list.append(text)
		
		#print(json.dumps(result, ensure_ascii=False), flush=True, file=ofp) ## json line 형식으로 저장
		return return_list


	def k2e(self,sent_list):

		patient = 0
		return_list=[]

		for line in sent_list:
			line = line.strip()
			try:
				text = self.translate(line,'ko','en') ## translatin
			except (KeyError,requests.exceptions.ConnectionError) as e:
				if patient > 5:
					ofp.close()
					exit() ## Error가 5번 이상 누적되면 종료
				patient += 1
				time.sleep(30) ## 에러 발생 시 1시간 대기
				continue

			return_list.append(text)
		
		#print(json.dumps(result, ensure_ascii=False), flush=True, file=ofp) ## json line 형식으로 저장
		return return_list



