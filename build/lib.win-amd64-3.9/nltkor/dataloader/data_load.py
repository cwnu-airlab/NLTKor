import json, pandas as pd
import numpy as np
import os
import csv
from pathlib import Path
from threading import Thread 
import GPUtil
import time
import torch, gc
import torch.nn as nn
from random import shuffle

class DataLoad :
    
    def __init__(self,data_type,batch_size,shuffle = False ,index= None,tem=None):
        self.extension = ["json","trn","jsonl"]
        self.df_dict = {}
        self.name_list = []
        self.data_type = data_type
        self.data_list = []
        self.index  = index
        self.batch_size  = batch_size
        self.monitor = Monitor(delay=0.001) 
        self.check_batch = 1
        self.used_memory_gpu_list = []
        self.data_memory_gpu = 1
        self.iter_count = 0
        self.save_batch = batch_size
        self.shuffle = shuffle
        if tem is None :
            self.tem = 10
        else :
            self.tem = tem
        self.current_position = 0
        self.sliced_data = pd.DataFrame()

    def __iter__ (self) :
        self.current_position = 0 
        return self 
    
    def __next__ (self) :
        # 현재 데이터프레임
        df = self.data_list
        if df is None:
            raise ValueError(f"{self.data_type} 데이터가 없습니다.")

        # 데이터프레임의 끝에 도달했는지 확인
        if self.current_position >= len(list(self.data_list.values())[0]):
            self.monitor.stop()
            raise StopIteration
        used_memory_gpu, total_memory_gpu = self.monitor.print_memory_utilization() 
        keys_list =  self.data_list.keys()
        result = []

        self.used_memory_gpu_list.append(used_memory_gpu)
        tem_mem = total_memory_gpu *(self.tem/100)
        if isinstance(self.batch_size,int) and (total_memory_gpu - used_memory_gpu) < tem_mem and len(self.used_memory_gpu_list) < 7 :
            self.batch_size = int(self.batch_size * 0.9)
            gc.collect()
            torch.cuda.empty_cache()
        elif self.iter_count > 3 and len(self.used_memory_gpu_list) < 10:
            if self.used_memory_gpu_list[-1] - self.used_memory_gpu_list[0] == 0 :
                memory_use = 1
            else :
                memory_use = self.used_memory_gpu_list[-1] - self.used_memory_gpu_list[0]
            batch_ = int((total_memory_gpu*(1-self.tem/100) - self.used_memory_gpu_list[0]) /(memory_use/self.batch_size))
            self.batch_size = batch_
            if self.batch_size > len(list(self.data_list.values())[0]) :
                self.batch_size = len(list(self.data_list.values())[0])
        if self.iter_count <= 3:
            for i in keys_list :
                result.append(self.data_list[i][self.current_position : self.current_position + self.check_batch])
            self.current_position += self.check_batch
            self.check_batch += 1
            if len(self.used_memory_gpu_list) == 3 :
                self.data_memory_gpu = self.used_memory_gpu_list[2] - self.used_memory_gpu_list[1]
                self.batch_size = self.check_batch
                self.monitor = Monitor(delay = 2)
                time.sleep(2)

        else :
            #self.stop_monitor() 
            if isinstance(self.save_batch,int) and self.save_batch < self.batch_size :
                self.batch_size = self.save_batch 
            
            for i in keys_list :
                result.append(self.data_list[i][self.current_position : self.current_position + self.batch_size])
            self.current_position += self.batch_size

        self.iter_count += 1
        return result

    def stop_monitor(self):
        if self.monitor:
            self.monitor.stop()
    
    def __len__ (self) :
        return len(list(self.data_list.values())[0])
                         
    def map (self,function) :
        self.data_list = list(map(function,self.data_list))
        
        
    def loader (self,root_dir) :
        dirc_pathname_list= os.listdir(root_dir)
        # os.walk()를 사용해 디렉토리 트리 탐색
        all_files = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                all_files.append(full_path)
                self.name_list.append(filename)
                
        for final_file_path in all_files:
            file_name = os.path.splitext(os.path.basename(final_file_path))[-2] 
            if "train" in file_name.lower() :
                data_key = "train"
            elif "valid" in file_name.lower() :
                data_key = "valid"
            elif "dev" in file_name.lower() :
                data_key = "valid"
            else :
                data_key = file_name
            
            # 확장자 확인
            if any(final_file_path.endswith(f".{ext}") for ext in self.extension):
                if final_file_path.endswith(".json"):
                    df = pd.read_json(f"{final_file_path}")
                    #print(df.head())
                   
                    
                elif final_file_path.endswith(".jsonl"):
                    df = pd.read_json(f"{final_file_path}",lines=True)
                    #print(df.head())
                  
                
                elif final_file_path.endswith(".trn"):
                    rows = []
                    csv_file_path = self.convert_to_csv(final_file_path,all_files)
                    
                    with open(csv_file_path, 'r', encoding='utf-8') as file:
                        for line in file:
                            parts = line.strip().split(", ", 1)  # 쉼표를 기준으로 나누되 최대 2개로 분할
                            if len(parts) == 2:
                                rows.append(parts)

                    # DataFrame으로 변환
                    df = pd.DataFrame(rows, columns=["FilePath", "Text"])
                    #print(df.head())

            else:
                print("지원하지 않는 확장자입니다.")
                continue
            self.df_dict[data_key] = df 
        self.data_list = self.df_dict[self.data_type]
        
        if self.index is not None :
            if type(self.index[0]) is str :
                self.data_index()
            elif type(self.index[0]) is tuple :
                self.data_slicing()
        
        
        self.data_list = { col : self.data_list[col].tolist() for col in self.data_list.columns}
        for key in self.data_list :
            shuffle(self.data_list[key])
    
    def data_slicing (self):
        slicing_pd = []
        for start, end in self.index:
            slicing_pd.append(self.data_list[start:end])  # iloc 슬라이싱   
        self.data_list = pd.concat(slicing_pd)

    def data_index (self): 
        self.index = list(set(self.index)) 
        df = self.df_dict[self.data_type]
        index_list = pd.DataFrame()
        for i in self.index :
            index_list = pd.concat([index_list, df.iloc[[i]]], ignore_index=True)
        self.data_list = index_list


    def convert_to_csv(self,input_file,folder_path):
        file_names = Path(input_file).stem 
        csv_folder_path = os.path.join(folder_path, "csv")

        if not os.path.exists(csv_folder_path):
            os.makedirs(csv_folder_path)

        output_file_path = os.path.join(csv_folder_path, f"{file_names}.csv")
 
        # 입력 파일을 읽고 CSV로 저장
        with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file_path, 'w', encoding='utf-8', newline='') as outfile:
            csv_writer = csv.writer(outfile)
            for line in infile:
                # "::"를 ","로 변경
                cleaned_line = line.replace("::", ",").strip()
                
                # 각 라인을 콤마로 나눠 리스트로 변환
                row = cleaned_line.split(",")
                
                # CSV 파일에 쓰기
                csv_writer.writerow(row)

        return output_file_path


class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
    
    def update_memory_utilization(self):
        gpus = GPUtil.getGPUs()  # Get all GPUs
        for gpu in gpus:
            self.memory_used[gpu.id] = gpu.memoryUsed
            self.memory_total[gpu.id] = gpu.memoryTotal
    
    def print_memory_utilization(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            device = torch.cuda.current_device()  # 현재 사용 중인 GPU
            properties = torch.cuda.get_device_properties(device)
            total_memory = properties.total_memory
            return reserved,total_memory



