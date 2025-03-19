import datetime
import os

def make_requirement(packages):
    file_path = os.path.abspath(__file__)
    file_path = file_path + "__requirement__NLTKor.txt"
    with open(file_path, "a") as f:
        for package in packages:
            f.write(package + '\n')

    return file_path