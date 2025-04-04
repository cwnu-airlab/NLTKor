from nltkor.dataloader.dataset.koreanvoice_ import KoreanVoice
from nltkor.dataloader.dataset.machinereading_ import MachineReading
from nltkor.dataloader.dataset.thesissummary_ import ThesisSummary
from nltkor.dataloader.dataset.documentsummary import DocumentSummary
from nltkor.dataloader.dataset.agriculture_ import Agriculture
from nltkor.dataloader.dataset.patent_ import Patent
from nltkor.dataloader.dataset.koreanconversationsummary_ import KoreanConversationSummary
from nltkor.dataloader.dataset.generalknowledge_ import GeneralKnowledge
from nltkor.dataloader.dataset.gigaword_ import GigaWord
from nltkor.dataloader.dataset.canard_ import CANARD
from nltkor.dataloader.data_load import DataLoad
from datasets import load_dataset
import os
from random import shuffle

class AirData : 
    
    def __init__(self):
        self.check = False
        self.dataset = None


    @property
    def data_list (self):
        return DATA_LIST

    def download(self,*args, root_dir=None, force_download=False):
        
        data_dict = DATA_DOWNLOAD
         
        for data in args :
            try :
                data_dict = data_dict[data]
            except :
                try :
                    self.dataset = load_dataset(data)
                    self.check = True
                except :
                    raise ValueError(f"Invalid category: {args}. Support only {DATA_LIST}")
        if self.check is False:
            data_dict.download_(root_dir)

    def get_data_loader (self,*args,data_type,batch_size,index = None, root_dir=None,tem = None) :
        
        data_name = []
        if self.check is False :
            for name in args :
                try :
                    data_name.append(DATA_LOAD[name])
                except :
                    raise ValueError(f"Invalid category: {args}. Support only {DATA_LIST}")

        if root_dir is None :     
            root_dir = os.getcwd()
        root_dir += f"/nltkor/dataloader/dataset/Data/"
        
        for path in data_name : 
            root_dir = os.path.join(root_dir,path)
            
        dataloader = DataLoad(data_type,batch_size,index,tem)
        if self.check is False:
            dataloader.loader(root_dir)
        else : 
            dataset_ = self.dataset[data_type]
            data_set = {key: [] for key in dataset_[0]}
            for item in dataset_:
                for key, value in item.items():
                    data_set[key].append(value) 
            for key in data_set :
                shuffle(data_set[key])
            dataloader.data_list = data_set
        #dataloader = DataLoad(root_dir,data_type,index = None)
        return dataloader
        

        

        
DATA_LIST = {
    '자연어 데이터' : ["한국어음성",
                    "기계독해",
                    {
                    "문서요약 텍스트": {
                        "법률",
                        "사설",
                        "신문기사"
                        }
                    }],
    'gigaword' : "gigaword",
    'CANARD' : "CANARD"
}

DATA_DOWNLOAD = {
    '자연어 데이터': {
        "한국어음성": KoreanVoice(),
        "기계독해" : MachineReading(),
        "문서요약 텍스트": {
                "법률": DocumentSummary("law"),
                "사설": DocumentSummary("editorial"),
                "신문기사": DocumentSummary("newspaper_article")
            }
    },
    'gigaword' : GigaWord(),
    'CANARD' : CANARD()
    
    }


DATA_LOAD = {
    '자연어 데이터' : "natural_language_data",
    "한국어음성" : "koreanvoice",
    "기계독해" : "machincereading",
    "문서요약 텍스트" : "document_summary",
    "법률" : "law",
    "사설" : "editorial",
    "신문기사" : "newspaper_article",
    "gigaword":"gigaword",
    "CANARD" : "CANARD"
    
}

if __name__ == '__main__':
   
    AirData().download("CANARD")
    df = AirData().get_data_loader("gigaword",data_type = "train",index = [0,3])
    print(df["train"].iloc[0])
