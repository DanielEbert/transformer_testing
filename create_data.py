import random
import os
import numpy as np

NUMBERS_PER_FILE = 100
GENERATE_COUNT = 100

updir = 'data/up'
downdir = 'data/down'

os.makedirs(updir, exist_ok=True)
os.makedirs(downdir, exist_ok=True)

def generate(out_dir, increment):
    for i in range(GENERATE_COUNT):
        numbers = []
        cur = random.uniform(0, 10)
        for _ in range(NUMBERS_PER_FILE):
            numbers.append(cur)
            cur += random.uniform(*increment)
        np.save(os.path.join(out_dir, f'{i}.npy'), np.array(numbers))

generate(updir, (-1, 5))
generate(downdir, (-5, 1))
