# Test run to see my manipulated data
# Using pandas libary to put my data in a row/column table to actually

from scipy.io import arff
import pandas as pd

data = arff.loadarff(r"C:\REU\REU_NMSU\final-dataset-short.arff")
df = pd.DataFrame(data[0])
print(df)

data.shape

from sklearn.ensemble import RandomForestClassifier