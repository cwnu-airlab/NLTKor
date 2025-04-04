import os
import requests
import zipfile
from nltkor.dataloader.data_download import DataDownload

class DocumentSummary(DataDownload) :

    def __init__(self,sub_category):
        self.sub_category = sub_category
        self.default_download_path = os.path.join(os.path.dirname(__file__), "Data")
        self.url_s = []

    def check_canard_(self) :
        """Checks if the data is available and downloads it if necessary"""
        if not os.path.exists(self.default_download_path+"/natural_language_data"):
                return False
        else:
                return True
            
    def download_(self, root_dir=None):
        if not self.check_canard_():
            if root_dir is None:
                root_dir = self.default_download_path
            else :
                root_dir = os.path.join(root_dir,"Data")
        
            # URL 생성
            base_url = f"https://air.changwon.ac.kr/~airdemo/storage/data/자연어 데이터/문서요약 텍스트"
        
            root_dir = os.path.join(root_dir,"natural_language_data")
            root_dir = os.path.join(root_dir,"document_summary") 
            root_dir = os.path.join(root_dir,self.sub_category)

            os.makedirs(root_dir, exist_ok=True)
            os.makedirs(root_dir+"/Validation", exist_ok=True)
            os.makedirs(root_dir+"/Training", exist_ok=True)
            
            self.url_s = [f"Validation/{self.sub_category}_valid_original.zip",f"Training/{self.sub_category}_train_original.zip"]
            
            for category_url in self.url_s :
                file_path = os.path.join(root_dir, category_url)

                print(f"Downloading...")
                with requests.get(os.path.join(base_url, category_url), stream=True) as r:
                        r.raise_for_status()
                        with open(file_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                                
                self.extract_and_delete_zip(file_path)
        else :
            print("already downloaded data")   
      


                    
if __name__ == '__main__':
    document_summary =DocumentSummary() 
    document_summary.download_() 
