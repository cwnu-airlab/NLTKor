from xml.etree.ElementTree import parse
import os, re
from operator import eq
import time
import nltkor
from nltkor.sejong.sejong_download import SejongDir


common_path=os.path.dirname(nltkor.sejong.__file__)
class Entry():

    def __init__(self, name, en, pos):
        self.name = name
        self.entry = en
        self.pos = pos
        self.SejongDir = SejongDir()
        

    def __repr__(self):
        return "%s('%s')" % (type(self).__name__, self.name)



    # sense객체 리턴
    def senses(self):
        list = []

        allsense = self.entry.findall("sense")
        for se in allsense:
            try:
                ss = str(self.name + "." + se.attrib['n'])
            except KeyError:
                ss = str(self.name)
            temp = Sense(ss, se, self.pos)
            list.append(temp)

        return list

    # 숙어
    def idm(self):
        list = []
        try:
            id = self.entry.find("idm_grp")
            idm = id.findall("idm")
        except AttributeError:
            return list

        for tmp in idm:
            if tmp.text is None:
                return list

            if '~' in tmp.text:
                name = self.name.split('.')
                tmp.text = tmp.text.replace('~', name[0])
            list.append(tmp.text)

        return list

    # 복합어
    def comp(self):

        list = []
        try:
            mor = self.entry.find("morph_grp")
            comp = mor.findall("comp")
        except AttributeError:
            return list

        for tmp in comp:
            if tmp.text is None:
                return list

            if '~' in tmp.text:
                name = self.name.split('.')
                tmp.text = tmp.text.replace('~', name[0])
            list.append(tmp.text)

        return list

    # 파생어
    def der(self):
        list = []
        try:
            mor = self.entry.find("morph_grp")
            comp = mor.findall("der")
        except AttributeError:
            return list
        for tmp in comp:
            if tmp.text is None:
                return list

            if '~' in tmp.text:
                name = self.name.split('.')
                tmp.text = tmp.text.replace('~', name[0])
            list.append(tmp.text)

        return list


class Sense():

    def __init__(self, name, se, pos):
        self.name = name
        self.sense = se
        self.pos = pos

    def __repr__(self):
        return "%s('%s')" % (type(self).__name__, self.name)

    # 공통 태그
    def common_lr(self, sense):
        sem = sense.find("sem_grp")
        lr = sem.find("lr")
        return lr

        # sem
    #sem
    def sem(self):
        list = []
        sem = self.sense.find("sem_grp")
        synn = sem.find("sem_class")
        try:
            synn = synn.text
        except AttributeError:
            return list

        list.append(synn)
        return list

        # if None in list:
        #     list = []
        #     return list
        # else:
        #     return list
    # 동의어
    def syn(self):
        list = []
        lr = self.common_lr(self.sense)
        try:
            synn = lr.findall("syn")
        except AttributeError:
            return list

        for tmp in synn:
            list.append(tmp.text)

        if None in list:
            list = []
            return list
        else:
            return list
    # 반의어
    def ant(self):
        list = []
        lr = self.common_lr(self.sense)
        try:
            ant = lr.findall("ant")
        except AttributeError:
            return list

        for tmp in ant:
            list.append(tmp.text)

        if None in list:
            list = []
            return list
        else:
            return list
    # 동위어
    def coord(self):
        list = []
        lr = self.common_lr(self.sense)
        try:
            coo = lr.findall("coord")
        except AttributeError:
            return list

        for tmp in coo:
            list.append(tmp.text)

        if None in list:
            list = []
            return list
        else:
            return list
    # 부분어
    def mero(self):
        list = []
        lr = self.common_lr(self.sense)
        try:
            me = lr.findall("mero")
        except AttributeError:
            return list

        for tmp in me:
            list.append(tmp.text)

        '''if not list:
            return("@@@@@",list)
        '''
        if None in list:
            list = []
            return list
        else:
            return list
    # 상위어
    def hyper(self):
        list = []
        lr = self.common_lr(self.sense)
        try:
            hy = lr.findall("hyper")
        except AttributeError:
            return list

        for tmp in hy:
            list.append(tmp.text)

        if None in list:
            list = []
            return list
        else:
            return list
    # 하위어
    def hypo(self):
        list = []
        lr = self.common_lr(self.sense)
        try:
            hy = lr.findall("hypo")
        except AttributeError:
            return list

        for tmp in hy:
            if '~' in tmp.text:
                name = self.name.split('.')
                tmp.text = tmp.text.replace('~', name[0])
                list.append(tmp.text)
            else:
                list.append(tmp.text)

        if None in list:
            list = []
            return list
        else:
            return list
    # 전체어
    def holo(self):
        list = []
        lr = self.common_lr(self.sense)
        try:
            ho = lr.findall("holo")
        except AttributeError:
            return list

        for tmp in ho:
            if '~' in tmp.text:
                name = self.name.split('.')
                tmp.text = tmp.text.replace('~', name[0])
                list.append(tmp.text)
            else:
                list.append(tmp.text)

        if None in list:
            list = []
            return list
        else:
            return list
    # 관련어
    def rel(self):
        list = []
        lr = self.common_lr(self.sense)
        try:
            rel = lr.findall("rel")
        except AttributeError:
            return list

        for tmp in rel:
            list.append(tmp.text)

        if None in list:
            list = []
            return list
        else:
            return list
    # 예시
    def example(self):
        list = []

        if self.pos != 'nng_s':
            return list

        else:
            sem = self.sense.find("sem_grp")
            eg = sem.findall("eg")
            for tmp in eg:
                if '~' in tmp.text:
                    name = self.name.split('.')
                    tmp.text = tmp.text.replace('~', name[0])
                    list.append(tmp.text)
                else:
                    list.append(tmp.text)

        if None in list:
            list = []
            return list
        else:
            return list
    # 영어
    def trans(self):
        list = []
        sem = self.sense.find("sem_grp")
        trs = sem.findall("trans")
        for tmp in trs:
            list.append(tmp.text)

        if None in list:
            list = []
            return list
        else:
            return list
    # 형용사 결합
    def comb_aj(self):
        list = []

        try:
            syn = self.sense.find("syn_grp")
            aj = syn.findall("comb_aj")
        except AttributeError:
            return list

        for tmp in aj:
            if tmp.text is None:
                return list

            if '~' in tmp.text:
                name = self.name.split('.')
                tmp.text = tmp.text.replace('~', name[0])
            list.append(tmp.text)

        return list
    # 명사 결합
    def comb_n(self):
        list = []
        try:
            syn = self.sense.find("syn_grp")
            n = syn.findall("comb_n")
        except AttributeError:
            return list
        for tmp in n:
            if tmp.text is None:
                return list

            if '~' in tmp.text:
                name = self.name.split('.')
                tmp.text = tmp.text.replace('~', name[0])
            list.append(tmp.text)

        return list
    # 동사 결합
    def comb_v(self):
        list = []
        try:
            syn = self.sense.find("syn_grp")
            v = syn.findall("comb_v")
        except AttributeError:
            return list

        for tmp in v:
            v = tmp.find("form").text
            if v is None:
                return list

            if '~' in v:
                name = self.name.split('.')
                v = v.replace('~', name[0])
            list.append(v)
        return list

    # frame
    def sel_rst(self):

        final = {}
        list = []

        if self.pos == 'nng_s':
            return list

        frame_grps = self.sense.findall("frame_grp")

        for grp in frame_grps:  # 각각의 frame_grp type
            sub_list = []
            for subsense in grp.findall('subsense'):  # n개의 subsense
                str = ""
                eg_list = []
                check = 0
                for sel_rst in subsense.findall('sel_rst'):  # m개의 sel_rst
                    check += 1
                    for tmp in sel_rst.attrib.items():

                        if (tmp[0] == 'arg'):
                            str += ("<" + tmp[0] + "=" + tmp[1] + " ")

                        if (tmp[0] == 'tht'):
                            str += (tmp[0] + "=" + tmp[1] + ">")
                    try:
                        str += (sel_rst.text)
                    except TypeError:
                        str += ' '

                    if (check != len(subsense.findall('sel_rst'))):
                        str += ', '

                for eg in subsense.findall('eg'):
                    eg_list.append(eg.text)

                sub_list.append(str)
                sub_list.append(eg_list)

            final[grp.find('frame').text] = sub_list

        return final

    # 최상위 경로
    def sem_path(self):

        cur_sem = self.sem()[0]
        if cur_sem == None:
            return []
        filename = common_path+'/dict_semClassNum.txt'
        with open(filename, 'r',encoding="cp949") as file_object:
            lines = file_object.read()

        #print(lines)
        temp_list = []
        sem_list = []
        str = ""

        # 리스트 형성
        for tmp in lines:
            if tmp != '\n' and tmp != '\t':
                str += tmp
            else:
                if (str != ''):
                    sem_list.append(str)
                str = ''

        # 입력 단어 sem 위치 찾기
        regex = re.compile(r"_" + cur_sem + '$')
        for x in sem_list:
            if regex.search(x):
                cur_sem = x
                temp_list.append(cur_sem)

        while len(cur_sem.split('_')[0]) > 1:

            if cur_sem.split('_')[0][-2] == '.':
                tmp = cur_sem.split('_')[0][0:-2] + '_'
            else:
                tmp = cur_sem.split('_')[0][0:-3] + '_'
            regex = re.compile(r"^" + tmp)

            for x in sem_list:
                if regex.search(x):
                    cur_sem = x
                    temp_list.append(x)

        return list(reversed(temp_list))


    #유사도
    def wup_similarity(self,target):
        #self sem
        sem = self.sense.find("sem_grp")
        synn = sem.find("sem_class")
        synn1 = synn.text


        #target sem
        sem=target.sense.find("sem_grp")
        synn=sem.find("sem_class")
        synn2=synn.text


        list=[]
        path=common_path+"/layer.txt"
        f=open(path,'r')
        lines=f.readlines()
        for tmp in lines:
            if '_'+synn1+'\n' in tmp:
                list.append(tmp)
            if '_'+synn2+'\n' in tmp:
                list.append(tmp)

        ch=[]
        for tmp in list:
            ch.append(tmp.split("_")[0])

        word1 =ch[0].split('.');
        word2 =ch[1].split('.');

        same=0

        for tmp in range (0, min(len(word1),len(word2))):
            if word1[tmp] == word2[tmp]:
                same+=2
            else:
             break

        if self.name==target.name:
            same+=2

        result=same/((len(word1)+len(word2))+2)

        return result



# sense 바로 접근
def sense(input):

    input_list = input.split('.')
    arg= (input_list[0]+'.'+input_list[1]+'.'+input_list[2])
    target =entry(arg)
    allsense =target.entry.findall("sense")

    for se in allsense:
        if input==str(target.name+'.'+se.attrib['n']):
            return Sense(input,se,target.pos)

    #ss = str(self.name + "." + se.attrib['n'])
    #ss = str(self.name)

# entry 바로 접근
def entry(input):

    input_list = input.split('.')
    path=common_path+""
    if 'nn' in input_list[1]:
        path += "/01. 체언_상세"
    elif input_list[1] == 'vv':
        path += "/02. 용언_상세//vv"
    elif input_list[1] == 'va':
        path += "/02. 용언_상세//va"
    else:
        return


    path += "//"+input_list[0]+".xml"

    tree = parse(path)
    root = tree.getroot()
    allentry = root.findall("entry")
    for en in allentry:
        try:
            if input==str(input_list[0]+"."+en.attrib['pos']+"." + en.attrib['n']):
                return Entry(str(input_list[0]+"."+en.attrib['pos']+"." + en.attrib['n']), en, str(en.attrib['pos']))
        except KeyError:
            if input==str(input_list[0]+"."+en.attrib['pos']):
                return Entry(str(input_list[0]+"."+en.attrib['pos']), en, str(en.attrib['pos']))

# entry 객체 리턴
def entrys(word):
    path = filecheck(word)
    list = []

    for tmp in path:
        tree = parse(tmp)
        root = tree.getroot()
        allentry = root.findall("entry")

        for en in allentry:
            try:
                es = str(word + "." + en.attrib['pos'] + "." + en.attrib['n'])
            except KeyError:
                es = str(word + "." + en.attrib['pos'])

            temp = Entry(es, en, str(en.attrib['pos']))
            list.append(temp)

    return list


def _syn(word):

    ets=entrys(word)
    syn_list=[]

    for et in ets:
        for se in et.senses():
            syn_list+=se.syn()

    return syn_list

'''
def entry_error():

    path="./02. 용언_상세//va"
    abs_dir=os.path.join(os.getcwd(),path)
    file_names=os.listdir(abs_dir)

    #print(file_names)
    #print(len(file_names))
    error_list=[]


    for word in file_names:

        fpath=path+"//"+word
        #print(fpath)
        tree=parse(fpath)
        root=tree.getroot()
        #print(root.findtext('orth'))
        allentry=root.findall("entry")

        for en in allentry:
            try:
                en.attrib['n']

            except:

                error_list.append(word)
                break;

    print(error_list)
    print(len(error_list))
    print(len(file_names))


    return error_list



def sense_error():

    path="./02. 용언_상세//va"
    abs_dir=os.path.join(os.getcwd(),path)
    file_names=os.listdir(abs_dir)

    error_list=[]


    for word in file_names:

        fpath=path+"//"+word
        tree=parse(fpath)
        root=tree.getroot()
        allentry=root.findall("entry")

        for en in allentry:
            allsense=en.findall("sense")
            for se in allsense:
                try:
                    se.attrib['n']
                except:

                    if word not in error_list:
                        error_list.append(word)
                    break;

    print(error_list)
    print(len(error_list))
    print(len(file_names))


    return error_list

'''

# file check
def filecheck(word):
    n_path = common_path+"/01. 체언_상세"
    vv_path = common_path+"/02. 용언_상세/vv"
    va_path = common_path+"/02. 용언_상세/va"


    path = [n_path, vv_path, va_path]
    ret_list = []
    check = word + ".xml"


    for tmp in path:
        if check in os.listdir(tmp):
            ret_list.append(tmp + "/" + check)

    return ret_list
