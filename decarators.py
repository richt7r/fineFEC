import numpy as np

def dump_decarator(func):
    def print_at_the_end():
        func()
        print(f'print_addeed')
    return print_at_the_end

