import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

print('Start ...')

CICID_DATA = pd.read_csv("Friday-WorkingHours-Morning.pcap_ISCX.csv")
#Strip whitespaces in column header names
CICID_DATA.rename(columns={c: c.strip() for c in CICID_DATA.columns.values.tolist()}, inplace=True) 
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 120)
print(CICID_DATA.dtypes)
print(CICID_DATA.shape)

# Need to convert Infinity to NaN
#CICID_DATA = CICID_DATA.select_dtypes(include=['object'])
CICID_DATA['Flow Bytes/s'] = pd.to_numeric(CICID_DATA['Flow Bytes/s'], errors='coerce')
CICID_DATA['Flow Packets/s'] = pd.to_numeric(CICID_DATA['Flow Packets/s'], errors='coerce')
#print(CICID_DATA.dtypes)

#Drop NaN rows - double check with 'CICID_DATA.isnull().any()'
CICID_DATA = CICID_DATA.dropna() 

X = CICID_DATA.drop('Label', axis=1)
y = CICID_DATA['Label']

#Preprocessing. Scaling and Unity-Based Normalization
X = preprocessing.scale(X)
X = preprocessing.normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


print('End of run')

# from sklearn.svm import SVC
# classifier = SVC(kernel='rbf', gamma="scale")
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

