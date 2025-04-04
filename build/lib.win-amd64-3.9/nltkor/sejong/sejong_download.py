import zipfile
import base64
import os
import requests

class SejongDir :
    
        def __init__(self, data_dir=None):
                    self.data_dir = data_dir
                    if data_dir == None:
                            path= os.path.join(os.path.dirname(__file__), 'sejong_dictionary')
                            self.data_dir = path
                        
                    self.path = ""

                    if not self._check_sejong_dictionary():
                        self._download_sejong_dictionary()

                    
        def _check_sejong_dictionary(self):
            """Checks if the sejong dictionary is available and downloads it if necessary"""
            if not os.path.exists(self.data_dir+"/sejong_dictionary.zip"):
                    return False
            else:
                    return True
                    
                    
        def _download_sejong_dictionary(self):
            """Downloads the sejong xml from the server"""
            temp_path = os.path.join(os.path.dirname(__file__),'sejong_dictionary.zip')
            url = "https://air.changwon.ac.kr/~airdemo/storage/sejong/sejong_dictionary.zip"
            print("Downloading sejong dictionary...")
            with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(temp_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                            # If you have chunk encoded response uncomment if
                            # and set chunk_size parameter to None.
                            #if chunk:
                                    f.write(chunk)
            self.unzip_with_correct_encoding(temp_path)
            
            
        def unzip_with_correct_encoding(self, zip_path):
            """Unzips a ZIP file and decodes filenames with proper encoding"""
            extract_to_dir = self.data_dir
 
            if not os.path.exists(extract_to_dir):
                os.makedirs(extract_to_dir)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    
                    if file.endswith('/'):

                        try:
                            decoded_dir = file.encode('cp437').decode('utf-8')
                        except Exception as e:
                            print(f"디렉토리명 디코딩 실패: {file} - {e}")
                            decoded_dir = file 

                        dir_path = os.path.join(extract_to_dir, "")
                        os.makedirs(dir_path, exist_ok=True)
                        continue
                    
                    try:
                        decoded_name = file.encode('cp437').decode('utf-8')
                    except Exception as e:
                        print(f"파일명 디코딩 실패: {file} - {e}")
                        decoded_name = file  

                    # 추출 경로 설정
                    extracted_path = os.path.join(extract_to_dir, decoded_name)
                    extracted_path = extracted_path.replace("sejong_xml_/", "")
                    # 파일 저장
                    with zip_ref.open(file) as original_file:
                        os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
                        with open(extracted_path, "wb") as f:
                            f.write(original_file.read())
            

                    

                    



