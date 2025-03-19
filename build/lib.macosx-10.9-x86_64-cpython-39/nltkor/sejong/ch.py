import os
import unicodedata
import sys


for filename in os.listdir('/01. 체언_상세//'):
	new_filename=filename.replcace(filename, unicodedata.normalize('NFD',filename))
	os.rename(filename, new_filename)




