import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import time
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from sklearn.tree import export_graphviz

print('Start ...')

#CICID_DATA = pd.read_csv("Friday-WorkingHours-Morning.pcap_ISCX.csv")
CICID_DATA = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")

#Strip whitespaces in column header names
CICID_DATA.rename(columns={c: c.strip() for c in CICID_DATA.columns.values.tolist()}, inplace=True) 
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 120)
#print(CICID_DATA.dtypes)
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


#Performance Evaluation w/o PCA
start_time = time.time()
classifier = RandomForestClassifier(max_depth=5, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
elapsed_time = time.time() - start_time
print(classification_report(y_test,y_pred))
print('Accuracy:' + str(accuracy_score(y_test, y_pred)))
print('Elapsed time: ' + str(elapsed_time))


# Extract single tree
estimator = classifier.estimators_[5]
CICID_DATA.feature_names = list(CICID_DATA)[:len(list(CICID_DATA))-1]

# Export as dot file
export_graphviz(estimator, out_file='tree_rf.dot', 
                feature_names = CICID_DATA.feature_names,
                rounded = True, proportion = False, filled = True, precision=10)

 

#PCA - Explore number of principal components to set using Explained Variance
# pca = PCA()
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
#print(pca.explained_variance_ratio_)

#Plotting the Cumulative Summation of the Explained Variance
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Variance') #for each component
# plt.title('Explained Variance')
#plt.show()

#Performance Evaluation against number of principal components
# for i in range(1,15):
pca = PCA(n_components=15)
X_train_temp = pca.fit_transform(X_train)
X_test_temp = pca.transform(X_test)

start_time = time.time()
classifier = RandomForestClassifier(max_depth=5, random_state=0) #default n_estimators=10 == 10 decision tree
classifier.fit(X_train_temp, y_train)
y_pred = classifier.predict(X_test_temp)
elapsed_time = time.time() - start_time
print(classification_report(y_test,y_pred))
print('Accuracy:' + str(accuracy_score(y_test, y_pred)))
print('Elapsed time: ' + str(elapsed_time))

# Extract single tree
estimator = classifier.estimators_[5]
CICID_DATA.feature_names = list(CICID_DATA)[:len(list(CICID_DATA))-1]

# Export as dot file
export_graphviz(estimator, out_file='tree_rf_pca.dot', 
                #feature_names = CICID_DATA.feature_names,
                rounded = True, proportion = False, filled = True, precision=10)

print('End of run')