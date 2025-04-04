import os
import requests
import zipfile

class DataDownload :
    
    def extract_and_delete_zip(self, zip_file_path):
        # ZIP 파일이 있는 디렉토리 경로
        root_dir = os.path.dirname(zip_file_path)
        folder_name = os.path.splitext(os.path.basename(zip_file_path))[0] 
        folder_path = os.path.join(root_dir, folder_name)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # ZIP 파일 압축 해제
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path) 
        
        # 서브 디렉토리 내 ZIP 파일도 처리
        os.remove(zip_file_path)
        self.handle_subdirectories(folder_path)

    def handle_subdirectories(self, root_dir):
   
        for dirpath, dirnames, filenames in os.walk(root_dir):
            
            for filename in filenames:
                if filename.endswith('.zip'):                    
                    zip_file_path = os.path.join(dirpath, filename)
                    self.extract_and_delete_zip(zip_file_path)

