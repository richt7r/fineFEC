import numpy as np
import networkx as nx
from pathlib import Path

def array_to_str(array: np.ndarray): return str(array).replace('\n','').strip('[] ')

def from_matlab_indices_to_python_indices(load_path: Path):
    file = open(str(load_path)).readlines()
    rows = np.zeros(len(file),dtype=int)
    cols = np.zeros(len(file),dtype=int)
    for i,line in enumerate(file):
        rows[i],cols[i] = int(line.strip('\n ').split(' ')[0]),int(line.strip('\n ').split(' ')[-1])
    H = np.zeros((rows.max(),cols.max()),dtype=int)
    H[rows-1,cols-1] = 1
    i1,i2 = np.where(H)
    sparce_file = open(f'{str(load_path.parent.joinpath(load_path.stem+'_python_ids'+load_path.suffix))}','w')
    np.set_printoptions(threshold=np.inf)
    sparce_file.write(f'{array_to_str(i1)}\n{array_to_str(i2)}\n')
    sparce_file.close()

def from_2d_array_to_indices(H: np.ndarray,save_file: Path):
    i1,i2 = np.where(H)
    sparce_file = open(save_file,'w')
    np.set_printoptions(threshold=np.inf)
    sparce_file.write(f'{array_to_str(i1)}\n{array_to_str(i2)}\n')
    sparce_file.close()

def from_indices_to_2d_array(h_path: Path):
    H_data = open(h_path).readlines()
    rows = np.array(H_data[0].strip('\n ').split(' '))
    cols = np.array(H_data[1].strip('\n ').split(' '))
    rows = rows[rows!=''].astype(int)
    cols = cols[cols!=''].astype(int)
    H = np.zeros((rows.max()+1,cols.max()+1),dtype=int)
    H[rows,cols] = 1
    return H

def from_indices_to_sparce(h_path: Path):
    H_data = open(h_path).readlines()
    rows = np.array(H_data[0].strip('\n ').split(' '))
    cols = np.array(H_data[1].strip('\n ').split(' '))
    rows = rows[rows!=''].astype(int)
    cols = cols[cols!=''].astype(int)
    H = np.zeros((rows.max()+1,cols.max()+1),dtype=int)
    H[rows,cols] = 1
    rows = H.shape[0]
    cols = H.shape[1]
    ones = np.sum(H)
    i1 = np.zeros(rows+1,dtype=int)
    i2 = np.zeros(ones,dtype=int)
    i = 0
    for r,row in enumerate(H):
        i1[r] = i
        for pos in np.where(row)[0]:
            i2[i] = pos
            i+=1
    i1[-1] = ones
    i3 = np.zeros(cols+1,dtype=int)
    i4 = np.zeros(ones,dtype=int)
    i = 0
    for c,col in enumerate(H.T):
        i3[c] = i
        for pos in np.where(i2 == c)[0]:
            i4[i] = pos
            i+=1
    i3[-1] = ones
    return i1,i2,i3,i4
