import re



def parse_morph(target):
		
	result=list()
	buf = target
	buf = re.sub(' ','',buf)
	loop = re.search("(\/([A-Za-z]+)(?:\+|\n|$))", buf)

	while loop:
		pos = buf.find(loop.group(1))
		spos = buf[:pos].rfind("\t") + 1
		result.append((buf[spos:pos], loop.group(2)))
		buf = buf[pos+len(loop.group(1)):]
		loop = re.search("(\/([A-Za-z]+)(?:\+|\n|$))",buf)
	return result




