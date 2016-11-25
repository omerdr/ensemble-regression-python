import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.2f}'.format
FILENAME = './final/misc/results.csv'

df = pd.read_csv(FILENAME)
# print(df.keys())
print(df.pivot_table(values=['MSE_train', 'MSE_val'],index='i',aggfunc=[np.mean, np.min, np.max]))
print(df.pivot_table(values=['MSE_train', 'MSE_val'],index='Unnamed: 0',aggfunc=[np.mean,np.min,np.max]))
