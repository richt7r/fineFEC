import numpy as np
import utils
from pathlib import Path
import time
import threading
from decoders.python import python_decoders
import formatters

def worker(config,data,error_counter,thread):
    if config['decoder'] == 'spa':
        data = python_decoders.spa(data,config['i1'],config['i2'],config['i3'],config['i4'],int(config['num_iterations']))
        error_counter[thread]+=np.where(utils.llr_to_bit(data))[0].size

np.random.seed(13)

print(f'\nsimulation config:\n')
config_path = Path('config.conf')
config_data = open(config_path).readlines()
config = {}
for line in config_data:
    print(line)
    param = line.split('=')
    config[param[0].strip(' ')] = param[1].strip(' \n')

i1,i2,i3,i4 = formatters.from_indices_to_sparce(Path(config['h_path']))
config['i1'], config['i2'], config['i3'], config['i4'] = i1,i2,i3,i4
n = i2.max()+1
m = i1.max()+1
k = n-m
start,stop,step = np.array(config['snr_range'].split(':'),dtype=float)
snrs = np.arange(start,stop,step)
print(f'\nsnr\t\t\tber\t\t\t\tsim_speed')
for snr in snrs:
    start_time = time.time()
    bits = 0
    errors = 0
    while errors < int(config['errors_to_point']):
        jobs = []
        error_counter = np.zeros(int(config['num_threads']),dtype=int)
        while jobs.__len__() < int(config['num_threads']):
            data = np.zeros(n,dtype=int)
            llrs = utils.gen_llr_bpsk(data,snr)            
            jobs.append(threading.Thread(target=worker,args=(config,llrs,error_counter,jobs.__len__())))
        for job in jobs:
            job.start()
        for job in jobs:
            job.join()    
        bits+= n*int(config['num_threads'])
        errors+= np.sum(error_counter)
        print(f'{snr:.2f}\t\t\t{(errors/bits):.5e}\t\t\t{(bits/(time.time()-start_time))/10**6:.5f} Mb/s',end='\r')
    print(f'{snr:.2f}\t\t\t{(errors/bits):.5e}\t\t\t{(bits/(time.time()-start_time))/10**6:.5f} Mb/s',end='\n')

        
