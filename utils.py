import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def gen_llr_bpsk(bits,snr,bounds = None):
    sigma = 10 ** (-snr/20)
    n = np.random.randn(bits.size)*sigma
    llrs = (2*((1-2*bits)+n))/(sigma**2)
    if bounds == None:
        return llrs
    else:
        return np.clip(np.round(llrs).astype(int),a_min=bounds[0],a_max=bounds[1])

def llr_to_bit(llrs):
    return (llrs<0).astype(int)

def plot(dumps: np.ndarray):
    for dump in dumps:
        file = open(dump).readlines()
        snr = []
        ber = []
        for line in file:
            if line[0].isdigit():
                data = np.array(line.strip('\n ').split(' '))
                snr.append(float(data[data!=''][0]))
                ber.append(float(data[data!=''][1]))
        plt.plot(snr,ber,label = Path(dump).stem)
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()
