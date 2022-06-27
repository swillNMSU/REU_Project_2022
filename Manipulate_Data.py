# Test run to see my manipulated data

from scipy.io import arff
import pandas as pd

data = arff.loadarff(r"C:\REU\REU_NMSU\final-dataset-short.arff")
df = pd.DataFrame(data[0])
print(df)