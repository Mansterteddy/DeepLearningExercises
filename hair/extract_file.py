# A helper function that transfer the file.

import os
import shutil

path = "/Users/zhangyuan/Documents/dataset/lfw/"
copy_path =  "/Users/zhangyuan/Documents/dataset/lfw_copy/"
#os.mkdir(copy_path)

for dirct in os.listdir(path):
    if dirct == '.DS_Store': continue
    dir_path = path + dirct
    for file in os.listdir(dir_path):
        if file == '.DS_Store': continue
        file_path = dir_path + '/' + file
        copy_file_path = copy_path + file
        shutil.copyfile(file_path, copy_file_path)
