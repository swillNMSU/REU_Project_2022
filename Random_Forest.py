# Manipulateing the data and utilizing random forest classifier to detect attack packets
# Using pandas libary to put my data in a row/column table to actually

from random import Random
from scipy.io import arff
import pandas as pd
import numpy as np

# Loads my data file into a variable
BaseData = arff.loadarff(r"C:\REU\REU_NMSU\final-dataset.arff")

# Creates a column and row table using pandas
data = pd.DataFrame(BaseData[0])

# Is a tuple of (row, columns)
data.shape

# Counts type of attack from data
# Short dataset: 20 Normal, 3 UDP FLood, 2 Smurf
data['PKT_CLASS'].value_counts()

# Prints each column title of data
data.columns

# New variable to only hold packet info not class of packet (normal or attack)
data_X = data.drop(['PKT_CLASS'], axis = 1)

# New Variable to hold type of class (normal or attack)
data_Y = data['PKT_CLASS']

# Finds unique type of packet class
# Short dataset: Normal, UDP Flood, Smurf
data_Y.unique()

'''
 Label Encoding: Convert the labels into a more machine-readable form (numerical)
'''

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Create packet classes into numbers and
# returns an array/list of each value
data_Y_trans = le.fit_transform(data_Y)

# Create packet types into numbers and
# returns an array/list of each value
le_PKT_TYPE = LabelEncoder()
le_PKT_TYPE.fit(data_X['PKT_TYPE'])
data_X['PKT_TYPE'] = le_PKT_TYPE.fit_transform(data_X['PKT_TYPE'])

# Create nodes_from into numbers and
# returns an array/list of each value
le_NODE_NAME_FROM = LabelEncoder()
le_NODE_NAME_FROM.fit(data_X['NODE_NAME_FROM'])
data_X['NODE_NAME_FROM'] = le_NODE_NAME_FROM.fit_transform(data_X['NODE_NAME_FROM'])

# Create nodes_to into numbers and
# returns an array/list of each value
le_NODE_NAME_TO = LabelEncoder()
le_NODE_NAME_TO.fit(data_X['NODE_NAME_TO'])
data_X['NODE_NAME_TO'] = le_NODE_NAME_TO.fit_transform(data_X['NODE_NAME_TO'])

# Create flags into numbers and
# returns an array/list of each value
le_FLAGS = LabelEncoder()
le_FLAGS.fit(data_X['FLAGS'])
data_X['FLAGS'] = le_FLAGS.fit_transform(data_X['FLAGS'])

'''
 Feature Selection: reducing the input variable 
 by using only relevant data and getting rid of noise in data.
'''

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(random_state = 42)
model.fit(data_X, data_Y_trans)

# Model utilizing tree structure to figure out most important feature
model.feature_importances_

# Creaeting a bar graph of most important features and only focused on first 20
feature_importance_std = pd.Series(model.feature_importances_, index = data_X.columns)

# New dataset for only the top 20 most important features
data_new_20features_X = data_X[['PKT_ID', 'TO_NODE', 'PKT_SIZE', 'FID', 'SEQ_NUMBER', 'NUMBER_OF_PKT', 'NUMBER_OF_BYTE', 'NODE_NAME_TO', 'PKT_IN', 'PKT_OUT', 'PKT_R', 'PKT_DELAY_NODE', 'PKT_RATE', 'BYTE_RATE', 'PKT_AVG_SIZE', 'UTILIZATION', 'PKT_DELAY', 'PKT_SEND_TIME', 'PKT_RESEVED_TIME', 'LAST_PKT_RESEVED']]

'''
Noraml Train Test Split: spliting the full data into a random training set and test sets
'''

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y_trans, test_size = 0.30, random_state = 42)

# Standarization Train Test Split: standardizng the train and test datasets to make sure data is normally distributed
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.fit_transform(X_test)

'''
Noraml Train Test Split: spliting the 20 feature selected data into a random training set and test sets
'''

X_train_20, X_test_20, Y_train_20, Y_test_20 = train_test_split(data_new_20features_X, data_Y_trans, test_size = 0.30, random_state = 42)

# Standarization Train Test Split: standardizng the train and test datasets to make sure data is normally distributed

ss_20 = StandardScaler()
X_train_std_20 = ss_20.fit_transform(X_train_20)
X_test_std_20 = ss_20.fit_transform(X_test_20)

'''
Random Forest: Creates multiple decision trees and chooses the best one
'''

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_std_20, Y_train_20)

rf_Y_pred = rf.predict(X_test_std_20)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("Classification Report for Random Forest: \n", classification_report(Y_test_20, rf_Y_pred))

rf_conf_mat = confusion_matrix(Y_test_20, rf_Y_pred)
print("Random Forest Confusion: \n", rf_conf_mat)

acc_score = accuracy_score(Y_test_20, rf_Y_pred)
print("Accuracy Score for Random Forest Classification: \n", acc_score * 100)