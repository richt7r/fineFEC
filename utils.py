import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

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
                data = line.split('\t')
                snr.append(float(data[0]))
                ber.append(float(data[3]))
        plt.plot(snr,ber,label = Path(dump).stem)
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()

def output(bits,errors,dump,config,snr,start_time):
    dump_file = open(config['dump_path'],'w')
    dump_file.write(dump + f'{snr:.2f}\t\t\t{(errors/bits):.5e}\t\t\t{(bits/(time.time()-start_time))/10**6:.5f} Mb/s\n')
    dump_file.close()
    if bool(config['terminal_output']): 
        print(f'{snr:.2f}\t\t\t{(errors/bits):.5e}\t\t\t{(bits/(time.time()-start_time))/10**6:.5f} Mb/s',end='\r')

def build_config(config_path: Path):
    config_data = open(config_path).readlines()
    config = {}
    config['config_path'] = config_path
    for line in config_data:
        param = line.split('=')
        config[param[0].strip(' ')] = param[1].strip(' \n')
    param_adv = [param == "" for param in config.values()]
    if sum(param_adv):
        missing_params = np.array(list(config.keys()))[np.where(param_adv)[0]]
        raise ValueError(
            f"following parameters not specified in {config_path}:\n{missing_params}"
        )
    return config

def build_header(config_path: Path):
    header =f'\nsimulation config:\n\n'
    config_data = open(config_path).readlines()
    for line in config_data:
        header+=line+'\n'
    header+=f'\nsnr\t\t\tber\t\t\t\tsim_speed\n'
    return header

