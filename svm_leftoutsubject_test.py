# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:46:02 2018

@author: sofha
"""


from sklearn import svm, metrics
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
accTrain=np.zeros((5,5))
acc=np.zeros((15,5,5))
accTest=np.zeros(15)
accTest_median=np.zeros(15)

accVal=np.zeros(15)
I1=np.zeros((15,2))
accTestPerm=np.zeros((15,100))
I1_median=np.zeros((15,2))
accTestPerm_median=np.zeros((15,100))
#%%

mat_contents = sio.loadmat('C:/Users/sofha/Documents/Mindreaders/Project-MindReading-master/Project-MindReading-master/matlab/EEG_events_600ms_ASR_100Hz_interp20Miss_zscore_avg.mat')
yAll=np.transpose(mat_contents['Animate_avg'])
XAll=mat_contents['EV_Z_avg']
test_sub=15
indTest=np.asarray(range(23*(test_sub-1),test_sub*23))
XTest=XAll[indTest,:]
yTest=yAll[indTest]

inds=np.asarray(range(345))
inds=np.delete(inds,(indTest),axis=0)
X=XAll[inds,:]
y=yAll[inds]
for val_sub in range(14):

    indVal=np.asarray(range(23*(val_sub),(val_sub+1)*23))
    XVal=X[indVal,:]
    yVal=y[indVal]
    indTrain=np.asarray(range(345-23))
    indTrain=np.delete(indTrain,(indVal),axis=0)
    XTrain=X[indTrain,:]
    yTrain=y[indTrain]
    
    Cspan=[0.5,1,1.5,10,15]
    gamma_span=[1/1500,1/1000,1/500,1/350,1/200]
    C_count=-1
    
    for Cc in Cspan:
        gam_count=-1
        C_count=C_count+1
        C_count
        for gam in gamma_span:
            gam_count=gam_count+1
            clf = svm.SVC(gamma=gam,C=Cc,random_state=1)#gamma='scale'
            clf.fit(XTrain, np.ravel(yTrain))  
            yVal_pred=clf.predict(XVal)
            #yTrain_pred=clf.predict(XTrain)
            #accTrain[C_count,gam_count]=metrics.accuracy_score(yTrain, yTrain_pred)
            acc[val_sub,C_count,gam_count]=metrics.accuracy_score(yVal, yVal_pred)
ACC=np.mean(acc,axis=0)       
i=np.argmax(ACC)
i1=np.unravel_index(i, (5,5))
I1[test_sub-1,:]=i1;
clf = svm.SVC(gamma=gamma_span[i1[1]],C=Cspan[i1[0]])#gamma='scale'
clf.fit(X, np.ravel(y)) 
yTest_pred=clf.predict(XTest)
#yVal_pred=clf.predict(XVal)
accTest[test_sub-1]=metrics.accuracy_score(yTest, yTest_pred)
for rep in range(100):
    rp=np.random.permutation(23)
    accTestPerm[test_sub-1,rep]=metrics.accuracy_score(yTest[rp], yTest_pred)
    
    
ACC=np.median(acc,axis=0)       
i=np.argmax(ACC)
i1=np.unravel_index(i, (5,5))
I1_median[test_sub-1,:]=i1;
clf = svm.SVC(gamma=gamma_span[i1[1]],C=Cspan[i1[0]])#gamma='scale'
clf.fit(X, np.ravel(y)) 
yTest_pred=clf.predict(XTest)
#yVal_pred=clf.predict(XVal)
accTest_median[test_sub-1]=metrics.accuracy_score(yTest, yTest_pred)
for rep in range(100):
    rp=np.random.permutation(23)
    accTestPerm_median[test_sub-1,rep]=metrics.accuracy_score(yTest[rp], yTest_pred)
#accVal[test_sub-1]=metrics.accuracy_score(yVal, yVal_pred)

#%%
perc_mean=np.zeros(15)
for sub in range(15):
    perc_mean[sub]=sum(accTest[sub]>accTestPerm[sub,:])
    
perc_median=np.zeros(15)
for sub in range(15):
    perc_median[sub]=sum(accTest_median[sub]>accTestPerm_median[sub,:])
 #%%
max_acc=np.zeros(15)
I2=np.zeros((15,2))
for test_sub in range(1,16):
    indTest=np.asarray(range(23*(test_sub-1),test_sub*23))
    indTrain=np.asarray(range(345))
    indTrain=np.delete(indTrain,(indTest),axis=0)
    XTrain=XAll[indTrain,:]
    yTrain=yAll[indTrain]
    XTest=XAll[indTest,:]
    yTest=yAll[indTest]
    accValTest=np.zeros((5,5))
    C_count=-1
    for Cc in Cspan:
            gam_count=-1
            C_count=C_count+1
            C_count
            for gam in gamma_span:
                gam_count=gam_count+1
                clf = svm.SVC(gamma=gam,C=Cc)#gamma='scale'
                clf.fit(XTrain, np.ravel(yTrain))  
                yTest_pred=clf.predict(XTest)
                #yTrain_pred=clf.predict(XTrain)
                #accTrain[C_count,gam_count]=metrics.accuracy_score(yTrain, yTrain_pred)
                accValTest[C_count,gam_count]=metrics.accuracy_score(yTest,yTest_pred)
    max_acc[test_sub-1]=np.max(accValTest)
    i=np.argmax(accValTest)
    i1=np.unravel_index(i, (5,5))
    I2[test_sub-1,:]=i1;

#%%
plt.figure(17)
plt.plot(accTest)
plt.plot(accTest_median)
plt.plot(max_acc)
plt.plot(np.zeros(15)+13/23)
plt.legend(('Parameters based on validation sets, mean','Parameters based on validation sets, median','Optimum parameters','Baseline'))
plt.plot(accTest,'ko')
plt.plot(max_acc,'kx')
plt.plot(perc_mean/100,color='#1f77b4',linestyle='dashed')
plt.plot(perc_median/100,color='#ff7f0e',linestyle='dashed')
plt.xlabel('Subjects')
plt.ylabel('Accuracy')
#plt.plot(yTest_pred)
#%%
Cs=np.asarray(Cspan)
Cval=np.asarray(I1[:,0], dtype=int)
Cmean=np.mean(Cs[Cval])

Gs=np.asarray(gamma_span)
Gval=np.asarray(I1[:,1], dtype=int)
Gmean=np.mean(Gs[Gval])