import pandas as pd
import numpy as np
from glob import iglob

TRAIN_LOSS = 0
TEST_LOSS  = 1
R2         = 2

DATA_DIR = '../data/DMproject/Results'

stuff = iglob(DATA_DIR)

best_r2 = (-1000,'')
for thing in stuff:
    print('thing: {}'.format(thing))
    models = iglob('{}/*'.format(thing))
    for model in models:
        this_best_R2 = -1000
        print('model: {}/*'.format(model))
        runs = iglob('{}/*'.format(model))
        for run in runs:
            print('run: {}/*'.format(run))
            data = pd.read_csv('{}'.format(run))
            data = data.as_matrix()
            data = np.array(data)
            max_r2 = np.amax(data[:,R2])
            if max_r2 > this_best_R2 : this_best_R2 = max_r2
        if this_best_R2 > best_r2[0] : best_r2 = (this_best_R2,model)
print("best_r2 : {}".format(best_r2))
        
