from time import time
import tracemalloc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn.preprocessing import KBinsDiscretizer

sns.set()
from sklearn import preprocessing

# data = pd.read_csv('car-evaluation.csv')
# data = pd.read_csv('compass-2016.csv')
# data = pd.read_csv('fico.csv')
# data = pd.read_csv('fico-binary.csv')
data = pd.read_csv('monk-binary.csv')
# data = pd.read_csv('tic-tac-toe.csv')
# data = pd.read_csv('balance-scale.csv')
# data = pd.read_csv('bar-7.csv')
# data = pd.read_csv('chudi.csv')
# data = pd.read_csv('iris.csv')
# newList = ['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','FM3Trans','fAlpha','fDist']

# newList = ['a1_2', 'a1_3', 'a1_4', 'a2_1', 'a2_2', 'a2_3', 'a2_4', 'a3_1', 'a3_2', 'a3_3', 'a3_4', 'a4_1', 'a4_2', 'a4_3', 'a4_4']
# newList =['f1','f2']

# newList = ['a1_high','a1_low','a1_med','a2_high','a2_low','a2_med','a3_2','a3_3','a3_4','a4_2','a4_4','a5_big','a5_med','a6_high','a6_low']
# newList = ['age','juv_crime','prior=0','prior=1','prior2-3','prior>3','two_year_recid']
# newList = ['PercentTradesWBalance','ExternalRiskEstimate','MSinceOldestTradeOpen','MSinceMostRecentTradeOpen','AverageMInFile','NumSatisfactoryTrades','NumTrades60Ever2DerogPubRec','NumTrades90Ever2DerogPubRec','PercentTradesNeverDelq','MSinceMostRecentDelq','MaxDelq2PublicRecLast12M','MaxDelqEver','NumTotalTrades','NumTradesOpeninLast12M','PercentInstallTrades','MSinceMostRecentInqexcl7days','NumInqLast6M','NumInqLast6Mexcl7days','NetFractionRevolvingBurden','NetFractionInstallBurden','NumRevolvingTradesWBalance','NumInstallTradesWBalance','NumBank2NatlTradesWHighUtilization']

# newList = ["ExternalRiskEstimate<0.49","ExternalRiskEstimate<0.65","ExternalRiskEstimate<0.80","NumSatisfactoryTrades<0.5","TradeOpenTime<0.6","TradeOpenTime<0.85","TradeFrequency<0.45","TradeFrequency<0.6","Delinquency<0.55","Delinquency<0.75","Installment<0.5","Installment<0.7","Inquiry<0.75","RevolvingBalance<0.4","RevolvingBalance<0.6","Utilization<0.6","TradeWBalance<0.33"]
newList = ['a1_1','a1_2','a2_1','a2_2','a3_1','a4_1','a4_2','a5_1','a5_2','a5_3','a6_1']
# newList = ['Feat0_o', 'Feat0_x', 'Feat1_o', 'Feat1_x', 'Feat2_o', 'Feat2_x', 'Feat3_o', 'Feat3_x', 'Feat4_o', 'Feat4_x','Feat5_o', 'Feat5_x', 'Feat6_o', 'Feat6_x', 'Feat7_o', 'Feat7_x', 'Feat8_o', 'Feat8_x']
# newList = ['passanger2','age0','age1','age2','age3','age4','age5','age6','Bar0','Bar1','Bar2','Bar3','Restaurant20to50>=4','direction_same']
# newList = ['sepal_length','sepal_width','petal_length','petal_width']
size = len(newList)
print(size)

# en = preprocessing.LabelEncoder()
# en.fit(['h','g'])
# data.loc[:,'class'] = en.transform(data['class'])


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import GOSDT

# discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
tdata = data
scaler = preprocessing.RobustScaler()
tdata[newList] = scaler.fit_transform(data[newList])
dlist = list(tdata)
train = dlist[0:size]
# print(train)
predict = dlist[-1]
X = tdata[train]
Y = tdata[[predict]]
# print(X)
# print(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
from sklearn.tree import DecisionTreeClassifier

# --------------------------------- #
# --------------------------------- #
# ----------    GOSDT      -------- #
# --------------------------------- #
# --------------------------------- #

tracemalloc.start()
startTime = time()
DT = GOSDT()
# DT = DecisionTreeClassifier(max_depth=size)
DT.fit(x_train, y_train)
DT_predicted = DT.predict(x_test)

endTime = time()
print("Time Taken by DST: ", endTime - startTime)
print("Memory occupied by DST:", tracemalloc.get_traced_memory())
tracemalloc.stop()
DTC = DT.score(x_test, y_test)
print('Classification Report of Decision Tree Induction:\n')
print("Accuracy of DT: ", DTC * 100, "%\n")
print(classification_report(y_test, DT_predicted))

dt_cf = confusion_matrix(y_test, DT_predicted)
sns.heatmap(dt_cf.T, square=True, annot=True, fmt='d', cbar=True)
plt.xlabel('true label')

plt.ylabel('predicted label')
plt.title('Confusion Matrix of DTC')
plt.show()

dt_fpr, dt_tpr, dt_threshold = metrics.roc_curve(y_test, DT_predicted)
dt_roc = metrics.auc(dt_fpr, dt_tpr)

plt.title('ROC graph for Decision Tree')
plt.plot(dt_fpr, dt_tpr, label='AUC = %0.2f' % dt_roc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

# --------------------------------- #
# --------------------------------- #
# ---------- Random Forest -------- #
# --------------------------------- #
# --------------------------------- #
from sklearn.ensemble import RandomForestClassifier

tracemalloc.start()
startTime = time()

RF = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True,bootstrap = True, max_features = 'sqrt')
RF.fit(x_train,y_train.values.ravel())
RF_predicted = RF.predict(x_test)

endTime = time()
print("Time Taken by Random Forest: ", endTime - startTime)
print("Memory occupied by Random Forest:", tracemalloc.get_traced_memory())
tracemalloc.stop()

RFS = RF.score(x_test,y_test.values.ravel())

print('Classification Report of RFC:\n')
print("\nAccuracy of RFC: ",RFS*100,"%\n")
print(classification_report(y_test, RF_predicted))

rf_cf = confusion_matrix(y_test, RF_predicted)
sns.heatmap(rf_cf.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title('Confusion Matrix of RFC');
plt.show()

rf_fpr, rf_tpr, rf_threshold = metrics.roc_curve(y_test, RF_predicted)
rf_roc = metrics.auc(rf_fpr, rf_tpr)
plt.title('ROC graph for Random Forest')
plt.plot(rf_fpr, rf_tpr, label='AUC = %0.2f' %rf_roc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

# --------------------------------- #
# --------------------------------- #
# ------------- SVM  -------------- #
# --------------------------------- #
# --------------------------------- #
data_scaled = data
scaler = preprocessing.StandardScaler()
data_scaled[newList] = scaler.fit_transform(data_scaled[newList])

dslist = list(data_scaled)
strain = dslist[0:size]
spredict = dslist[-1]
Xs = data_scaled[strain]
Ys = data_scaled[[spredict]]
# print(Xs)
# print(Ys)
xs_train, xs_test, ys_train, ys_test = train_test_split(Xs,Ys,test_size=0.3)

from sklearn.svm import LinearSVC

tracemalloc.start()
startTime = time()

SV = LinearSVC(max_iter = 30000)
SV.fit(x_train,y_train.values.ravel())
SV_predicted = SV.predict(x_test)

endTime = time()
print("Time Taken by SVM: ", endTime - startTime)
print("Memory occupied by SVM:", tracemalloc.get_traced_memory())
tracemalloc.stop()

SVS = SV.score(x_test,y_test.values.ravel())
print('Classification Report of SVM:\n')
print("\nAccuracy of SVC: ",SVS*100,"%\n")
print(classification_report(y_test, SV_predicted))
sv_cf = confusion_matrix(y_test,SV_predicted)
print(sv_cf)

sns.heatmap(sv_cf.T, square=True, annot=True, fmt='d', cbar=True)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title('Confusion Matrix of SVM');
plt.show()

sv_fpr, sv_tpr, sv_threshold = metrics.roc_curve(y_test, SV_predicted)
sv_roc = metrics.auc(sv_fpr, sv_tpr)
plt.title('ROC graph for SVM')
plt.plot(sv_fpr, sv_tpr, label='AUC = %0.2f' %sv_roc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

# --------------------------------- #
# --------------------------------- #
# ------- Naive Bayesian ---------- #
# --------------------------------- #
# --------------------------------- #

data_norm = data
normalizer = preprocessing.Normalizer()
# normalizer=preprocessing.RobustScaler()
data_norm[newList] = normalizer.fit_transform(data_norm[newList])
dnlist = list(data_norm)
ntrain = dnlist[0:size]
npredict = dlist[-1]
Xn = data_norm[ntrain]
Yn = data_norm[[npredict]]
xn_train, xn_test, yn_train, yn_test = train_test_split(Xn,Yn,test_size=0.2)
from sklearn.naive_bayes import GaussianNB
tracemalloc.start()
startTime = time()

NB = GaussianNB()
NB.fit(x_train,y_train.values.ravel())
NB_predicted = NB.predict(x_test)

endTime = time()
print("Time Taken by Naive Bayesian: ", endTime - startTime)
print("Memory occupied by Naive Bayesian:", tracemalloc.get_traced_memory())
tracemalloc.stop()

NBS = NB.score(x_test,y_test.values.ravel())
print('Classification Report of Naive Bayes:\n')
print("\nAccuracy of Naive Bayes: ",NBS*100,"%\n")
print(classification_report(y_test, NB_predicted))
nb_cf = confusion_matrix(y_test,NB_predicted)
sns.heatmap(nb_cf.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title('Confusion Matrix of Naive Bayes');
plt.show()

nb_fpr, nb_tpr, nb_threshold = metrics.roc_curve(y_test, NB_predicted)
nb_roc = metrics.auc(nb_fpr, nb_tpr)
plt.title('ROC graph for Naive Bayes')
plt.plot(nb_fpr, nb_tpr, label='AUC = %0.2f' %nb_roc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

