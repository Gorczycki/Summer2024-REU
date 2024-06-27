import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import iisignature
import pandas as pd

df1 = pd.read_csv('Exxondata2 copy.csv')
close_prices1 = df1['close'].values

df2 = pd.read_csv('Shelldata2 copy.csv')
close_prices2 = df2['close'].values 


path1 = np.array([close_prices1, close_prices2])

signature = iisignature.sig(path1,2)
print(signature)
