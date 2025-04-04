import os
import requests
from urllib.parse import quote
import zipfile
from nltkor.dataloader.data_download import DataDownload

class MachineReading(DataDownload) :
    
    def __init__(self):
        self.default_download_path = os.path.join(os.path.dirname(__file__), "Data")
        self.check = 0
        
    def check_canard_(self) :
        """Checks if the data is available and downloads it if necessary"""
        if not os.path.exists(self.default_download_path+"/natural_language_data"):
                return False
        else:
                return True
    
    def download_(self,root_dir = None):
        if not self.check_canard_():         
            if root_dir is None:
                root_dir = self.default_download_path
            base_url = "https://air.changwon.ac.kr/~airdemo/storage/data/자연어 데이터/machincereading.zip"
            print("Downloading...")
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            
            # 다운로드 파일 경로
            file_path = os.path.join(root_dir, "natural_language_data.zip")
            with requests.get(base_url, stream=True) as r:
                        r.raise_for_status()
                        with open(file_path, 'wb') as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                # If you have chunk encoded response uncomment if
                                # and set chunk_size parameter to None.
                                #if chunk:
                                        f.write(chunk)
            
            self.extract_and_delete_zip(file_path)
        else :
            print("already downloaded data")


if __name__ == '__main__':
    machine_reading = MachineReading() 
    machine_reading.download_()  
