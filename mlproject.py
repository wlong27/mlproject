import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

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
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('Accuracy:' + str(accuracy_score(y_test, y_pred)))

#PCA - Explore number of principal components to set using Explained Variance
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(pca.explained_variance_ratio_)

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance') #for each component
plt.title('Explained Variance')
plt.show()

#Performance Evaluation against number of principal components
for i in range(1,15):
    pca = PCA(n_components=i)
    X_train_temp = pca.fit_transform(X_train)
    X_test_temp = pca.transform(X_test)

    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train_temp, y_train)
    y_pred = classifier.predict(X_test_temp)

    # print(confusion_matrix(y_test,y_pred))
    # print(classification_report(y_test,y_pred))
    print( str(i) + ' Accuracy:' + str(accuracy_score(y_test, y_pred)))
print('End of run')
