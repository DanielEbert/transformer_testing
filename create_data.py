import random
import os
import numpy as np

NUMBERS_PER_FILE = 100
GENERATE_COUNT = 100

updir = 'data/up'
downdir = 'data/down'

os.makedirs(updir, exist_ok=True)
os.makedirs(downdir, exist_ok=True)

# To create static information (same at all timestamps), we can add a [temperature, 0 , ..., 0] vector

def generate(out_dir, increment):
    for i in range(GENERATE_COUNT):
        randoms = []
        numbers = []
        cur = random.uniform(0, 10)
        for _ in range(NUMBERS_PER_FILE):
            numbers.append(cur)
            cur += random.uniform(*increment)

            randoms.append(random.uniform(0, 1))
        np.save(os.path.join(out_dir, f'{i}.npy'), np.array((numbers, randoms)))

generate(updir, (-4, 5))
generate(downdir, (-5, 4))
