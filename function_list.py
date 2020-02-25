import os
import glob as gb
import glob
import numpy as np

# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(gb.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F

def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)