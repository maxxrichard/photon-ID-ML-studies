#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import pandas as pd
import ROOT
import root_numpy as rootnp
import array

#from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Dropout,AlphaDropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# In[2]:


fname_sig = '/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/fullRun2/FinalNtuples/PyPt17_inf_mc16a_p3931_Rel21_AB21.2.94_v0.root'
fname_bkg = '/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/fullRun2/FinalNtuples/Py8_jetjet_mc16a_p3929_Rel21_AB21.2.94_v0.root'


# In[3]:


list_of_branches_ShowerShapes=['y_Reta',
                  'y_Rphi',
                  'y_weta1',
                  'y_weta2',
                  'y_deltae',
                  'y_fracs1',
                  'y_Eratio',
                  'y_wtots1',
                  'y_Rhad',
                  'y_Rhad1',
                  'y_f1',
                  'y_e277']

list_of_branches_Isolations=['y_ptcone20',
                  'y_ptcone40',
                  'y_topoetcone20',
                  'y_topoetcone40']

list_of_branches_for_binning=['y_eta',
                        'y_pt',
                        'evt_mu',
                        'y_convType']

list_of_branches_for_selection=['y_isTruthMatchedPhoton', 'acceptEventPtBin','y_IsLoose','y_IsTight']

weight_branch=['mcTotWeight']

list_of_branches = list_of_branches_ShowerShapes+list_of_branches_Isolations+list_of_branches_for_binning
list_of_branches = list_of_branches+list_of_branches_for_selection+weight_branch

list_of_branches_train = list_of_branches_ShowerShapes


# In[4]:


selection_sig = 'y_pt>25 && (y_eta<2.37) && y_convType==0 && (y_eta<1.37 || y_eta>1.52) && y_isTruthMatchedPhoton'
selection_bkg = 'y_pt>25 && (y_eta<2.37) && y_convType==0 && (y_eta<1.37 || y_eta>1.52) && !y_isTruthMatchedPhoton'

selection_data = ''

file_sig,file_bkg = ROOT.TFile(fname_sig),ROOT.TFile(fname_bkg)
tree_sig,tree_bkg = file_sig.Get('SinglePhoton'), file_bkg.Get('SinglePhoton')

nEvents_sig = None
nEvents_bkg = None

data_sig = rootnp.tree2array(tree_sig, branches=list_of_branches, 
                        selection=selection_sig,
                        start=0, stop=nEvents_sig)
data_bkg = rootnp.tree2array(tree_bkg, branches=list_of_branches, 
                        selection=selection_bkg,
                        start=0, stop=nEvents_bkg)

weights_sig = rootnp.tree2array(tree_sig, branches=weight_branch, selection=selection_sig, start=0, stop=nEvents_sig)
weights_bkg = rootnp.tree2array(tree_bkg, branches=weight_branch, selection=selection_bkg, start=0, stop=nEvents_bkg)


# In[5]:


# Labeling Data
df_sig = pd.DataFrame(data_sig, columns=list_of_branches, dtype=np.float32)
df_sig['class']=1 # indicator of signal sample

df_bkg = pd.DataFrame(data_bkg, columns=list_of_branches, dtype=np.float32)
df_bkg['class']=0 # indicator of signal sample

# Add two samples
df_balanced = df_sig.append(df_bkg, ignore_index=True)
#print(df_sig.tail(10))


# In[1]:


import lightgbm


# In[ ]:


wsum_sig = df_balanced[df_balanced['class']==1]['mcTotWeight'].sum()
wsum_bkg = df_balanced[df_balanced['class']==0]['mcTotWeight'].sum()
print('Sum of weights sig/bkg = ',wsum_sig,'/', wsum_bkg,'factor of ', 1./wsum_sig*wsum_bkg)
df_balanced['weight_nominal_abs_corr'] = df_balanced.apply(lambda row: 1./wsum_sig*wsum_bkg*row['mcTotWeight'] if row['class']==1 else row['mcTotWeight'], axis=1)
print('Sum of weights after rebalancing ',df_balanced[df_balanced['class']==1]['weight_nominal_abs_corr'].sum(),'/', df_balanced[df_balanced['class']==0]['weight_nominal_abs_corr'].sum())


# In[ ]:


train_frac = 0.7

# Splitting into train/test
df_sig_train, df_sig_test = train_test_split(df_sig, train_size=train_frac, shuffle=True)
df_bkg_train, df_bkg_test = train_test_split(df_bkg, train_size=train_frac, shuffle=True)


# In[6]:



df_train = df_sig_train.append(df_bkg_train, ignore_index=True)
df_test = df_sig_test.append(df_bkg_test, ignore_index=True)


X_train, X_test = df_train[list_of_branches_train], df_test[list_of_branches_train]
Y_train, Y_test = df_train['class'], df_test['class']

w_train, w_test = df_train['weight_nominal_abs_corr'], df_test['weight_nominal_abs_corr']

    
## Scale Inputs
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

print(pd.DataFrame(X_train, columns=list_of_branches_train, dtype=np.float32))
print('input shape: ', df_train[list_of_branches_train].shape)
num_features = df_train[list_of_branches_train].shape[1]
print('number of inputs: ',num_features)
input_shape = (num_features,)


# In[7]:


def classifier_model(input_shape):
    # Input
    X_input = Input(input_shape)
    
    # Layer(s)
    X = AlphaDropout(rate=0.12)(X_input)
    X = Dense(32, activation="selu", 
              kernel_initializer='lecun_normal', # he_normal, he_uniform, lecun_uniform
              name = 'Dense1')(X)

    X = AlphaDropout(rate=0.10)(X)
    X = Dense(64, activation="selu", 
              kernel_initializer='lecun_normal',
              name = 'Dense2')(X)
    
    X = AlphaDropout(rate=0.10)(X)
    X = Dense(32, activation="selu", 
              kernel_initializer='lecun_normal',
              name = 'Dense3')(X)
    
    # Output
    X_output = Dense(1, activation='sigmoid', name='output_layer')(X)
    
    # Build model
    model = Model(inputs=X_input, outputs=X_output, name='classifier_model')

    return model


# In[ ]:


nn_classifier_model = classifier_model((num_features,))
nn_classifier_model.compile(optimizer = "nadam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Learning Rate Performance Scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7)

history = nn_classifier_model.fit(x=X_train, y=Y_train.values, 
                                  sample_weight=w_train.values, 
                                  validation_data = (X_test,Y_test.values, w_test.values),
                                  epochs=5, batch_size=64) # callbacks=[lr_scheduler]


# In[ ]:


plt.figure(1, figsize=(8, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid(True)
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('accuracy.png')


# In[ ]:


# ROC curve
y_pred_train = nn_classifier_model.predict(X_train).ravel()
fpr_train, tpr_train, thresholds_train = roc_curve(Y_train, y_pred_train, sample_weight=w_train.values)
auc_train = auc(fpr_train, tpr_train)

y_pred_test = nn_classifier_model.predict(X_test).ravel()
fpr_test, tpr_test, thresholds_test = roc_curve(Y_test, y_pred_test, sample_weight=w_test.values)
auc_test = auc(fpr_test, tpr_test)

plt.figure(3)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_train, tpr_train, label='TF2.0 train (area = {:.3f})'.format(auc_train))
plt.plot(fpr_test, tpr_test, label='TF2.0 test (area = {:.3f})'.format(auc_test))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
#plt.show()
plt.savefig('roc.png')


# In[ ]:


def plot_score(y_pred_sig, y_pred_bkg, w_sig, w_bkg,
              y_pred_sig2, y_pred_bkg2, w_sig2, w_bkg2, label):
    n, bins, patche = plt.hist( [y_pred_sig, y_pred_bkg], bins=25,color=["blue","red"], weights=[w_sig,w_bkg],
                               alpha = 0.5, 
                               label=[label[0]+' train',label[1]+' train'], histtype='stepfilled', range=(0.,1.) )

    plt.hist( [y_pred_sig2, y_pred_bkg2], bins=25,color=["blue","red"], weights=[w_sig2,w_bkg2],
                               alpha = 0.5, 
                               label=[label[0]+' test',label[1]+' test'], histtype='step', range=(0.,1.),linewidth=2.5 )
    plt.xlim(0, 1)
    plt.legend(loc='upper right')
    plt.ylabel('Events')
    plt.xlabel('NN Score');
    plt.grid()
    #plt.show()
    plt.savefig('score.png')

pd_pred_train = pd.DataFrame(y_pred_train,columns=["pred"], dtype=np.float32)
pd_pred_train['Y']=Y_train.values
pd_pred_train['weight']=w_train.values

pd_pred_test = pd.DataFrame(y_pred_test,columns=["pred"], dtype=np.float32)
pd_pred_test['Y']=Y_test.values
pd_pred_test['weight']=w_test.values*train_frac/(1.-train_frac)

plt.figure(4)
plot_score(pd_pred_train[pd_pred_train.Y==1].pred, pd_pred_train[pd_pred_train.Y==0].pred,
            pd_pred_train[pd_pred_train.Y==1].weight, pd_pred_train[pd_pred_train.Y==0].weight,
            pd_pred_test[pd_pred_test.Y==1].pred, pd_pred_test[pd_pred_test.Y==0].pred,
            pd_pred_test[pd_pred_test.Y==1].weight, pd_pred_test[pd_pred_test.Y==0].weight, [r'$tHq$',r'$t\bar{t}$'])


# In[ ]:




