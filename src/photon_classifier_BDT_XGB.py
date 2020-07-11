import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import ROOT
import root_numpy as rootnp
import array

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score,confusion_matrix,accuracy_score

import xgboost as xgb
import lightgbm as lgbm
from sklearn.model_selection import GridSearchCV

channel = 'lephad' # hadhad or lephad

#np.random.seed(7)

# Input files
fname_sig = '../tH_nominal_1l2tau_'+channel+'_012b_Xj/tHq/tHq.root'
fname_bkg = '../tH_nominal_1l2tau_'+channel+'_012b_Xj/ttbar/ttbar.root'
fname_data = '../tH_nominal_1l2tau_'+channel+'_012b_Xj/data/data.root'

# Some bool setup variables
is_abs_weight = True
do_one_hot_encoding = False
do_update_ntuples = True
do_GridSearch = False

train_frac = 0.7

# Variables
list_of_branches_leptons=['pt_lep1','eta_lep1','charge_lep1','type_lep1',
                          'pt_lep2','eta_lep2','charge_lep2','type_lep2',
                          'pt_lep3','eta_lep3','charge_lep3','type_lep3' ]

list_of_branches_jets=['pt_jet1','eta_jet1','btagged_jet1',
                       'pt_jet2','eta_jet2','btagged_jet2',
                       'pt_jet3','eta_jet3','btagged_jet3']               

list_of_branches_kinematics=['m_deltaPhiTau',
                  'm_deltaRTau',
                  'm_HvisMass',
                  'm_HvisPt',
                  'm_TvisMass',
                  'm_TvisPt',
                  'm_met_W',
                  'm_met_diTau']

if channel=='lephad':
    list_of_branches_kinematics = ['m_met', 'm_sumet', 'm_vis_H_Mass', 'm_vis_H_Pt', 'm_vis_Top_Mass', 'm_vis_Top_Pt',
                        'm_SS_ee','m_SS_emu','m_SS_mue','m_SS_mumu', 'm_OS_ee','m_OS_emu','m_OS_mue','m_OS_mumu']

list_of_branches_njets=['m_njets','m_nbjets']
weight_branch=['weight_nominal']

# All variables
list_of_branches = list_of_branches_njets+list_of_branches_kinematics+weight_branch
list_of_branches = list_of_branches+list_of_branches_leptons+list_of_branches_jets

# Training variables
list_of_branches_train = list_of_branches_njets+list_of_branches_kinematics
list_of_branches_train = list_of_branches_train+list_of_branches_leptons+list_of_branches_jets

# Reading trees
#selection_sig = 'm_nbjets==1'
#selection_bkg = 'm_nbjets==1'
#selection_data = 'm_nbjets==1'
selection_sig = ''
selection_bkg = ''
selection_data = ''

file_sig,file_bkg = ROOT.TFile(fname_sig),ROOT.TFile(fname_bkg)
tree_sig,tree_bkg = file_sig.Get('tHqLoop_nominal'), file_bkg.Get('tHqLoop_nominal')

nEvents_sig = None # 5000000
nEvents_bkg = None # None

data_sig = rootnp.tree2array(tree_sig, branches=list_of_branches, 
                        selection=selection_sig,
                        start=0, stop=nEvents_sig)
data_bkg = rootnp.tree2array(tree_bkg, branches=list_of_branches, 
                        selection=selection_bkg,
                        start=0, stop=nEvents_bkg)

weights_sig = rootnp.tree2array(tree_sig, branches=weight_branch, selection=selection_sig, start=0, stop=nEvents_sig)
weights_bkg = rootnp.tree2array(tree_bkg, branches=weight_branch, selection=selection_bkg, start=0, stop=nEvents_bkg)

# Labeling Data
df_sig = pd.DataFrame(data_sig, columns=list_of_branches, dtype=np.float32)
df_sig['class']=1 # indicator of signal sample
df_bkg = pd.DataFrame(data_bkg, columns=list_of_branches, dtype=np.float32)
df_bkg['class']=0 # indicator of signal sample


# Splitting into train/test
df_sig_train, df_sig_test = train_test_split(df_sig, train_size=train_frac, shuffle=True)
df_bkg_train, df_bkg_test = train_test_split(df_bkg, train_size=train_frac, shuffle=True)

# Checking for balance
print("Balancing samples...")
nevents_sig = df_sig.index[-1]
nevents_bkg = df_bkg.index[-1]
sig_corr_factor,_ = divmod(nevents_bkg/nevents_sig,1)
print ('Number of sig/bkg events before cutting = ',nevents_sig,'/', nevents_bkg,'( factor of ',nevents_bkg/nevents_sig,')')

print('train/test sig: ', df_sig_train.shape[0], '/', df_sig_test.shape[0])
print('train/test bkg: ', df_bkg_train.shape[0], '/', df_bkg_test.shape[0])

# Creating new data frame
def add_events(df_sig, df_bkg):
    df_balanced = df_sig.copy()
    df_balanced = df_balanced.append(df_bkg, ignore_index=True)
    return df_balanced.sample(frac=1)

df_balanced_train = add_events(df_sig_train, df_bkg_train)
df_balanced_test = add_events(df_sig_test, df_bkg_test)

print(df_balanced_train.head(20))
print('After rebalancing we have the following total number of train events: ',df_balanced_train.shape[0])
print('After rebalancing we have the following total number of test events: ',df_balanced_test.shape[0])

# Balance of weights
print("Balancing weights...")

def do_balance_weights(df_balanced, is_abs_weight):
    # Make abs value of weights
    df_balanced['weight_nominal_abs'] = df_balanced.apply(lambda row: -1*row['weight_nominal'] if row['weight_nominal']<0 else row['weight_nominal'], axis=1) 
    if(is_abs_weight):
        wsum_sig = df_balanced[df_balanced['class']==1]['weight_nominal_abs'].sum()
        wsum_bkg = df_balanced[df_balanced['class']==0]['weight_nominal_abs'].sum()
        print('Sum of weights sig/bkg = ',wsum_sig,'/', wsum_bkg,'factor of ', 1./wsum_sig*wsum_bkg)
        df_balanced['weight_nominal_abs_corr'] = df_balanced.apply(lambda row: 1./wsum_sig*wsum_bkg*row['weight_nominal_abs'] if row['class']==1 else row['weight_nominal_abs'], axis=1)
        print('Sum of weights after rebalancing ',df_balanced[df_balanced['class']==1]['weight_nominal_abs_corr'].sum(),'/', df_balanced[df_balanced['class']==0]['weight_nominal_abs_corr'].sum())
    else:
        wsum_sig = df_balanced[df_balanced['class']==1]['weight_nominal'].sum()
        wsum_bkg = df_balanced[df_balanced['class']==0]['weight_nominal'].sum()
        print('Sum of weights sig/bkg = ',wsum_sig,'/', wsum_bkg,'factor of ', 1./wsum_sig*wsum_bkg)
        df_balanced['weight_nominal_corr'] = df_balanced.apply(lambda row: 1./wsum_sig*wsum_bkg*row['weight_nominal_abs'] if row['class']==1 else row['weight_nominal_abs'], axis=1)
    return df_balanced

print('Training sample:')
df_balanced_train = do_balance_weights(df_balanced_train, is_abs_weight)
print('Testing sample:')
df_balanced_test = do_balance_weights(df_balanced_test, is_abs_weight)

# One Hot Encoding
def do_one_hot_encode_nJets(df_balanced, varname, list_of_branches_train, encoder, is_transform):
    var_encoded = None
    if(not is_transform):
        encoder = OneHotEncoder(categories='auto')
        var_encoded = encoder.fit_transform(df_balanced[varname].values.reshape(-1,1)).toarray()
    else:
        var_encoded = encoder.transform(df_balanced[varname].values.reshape(-1,1)).toarray()

    # update data frame
    list_of_branches_encoded = [varname+'_'+str(int(i)) for i in range(var_encoded.shape[1])]
    df_encoded = pd.DataFrame(var_encoded, columns = list_of_branches_encoded, dtype=np.float32)
    df_balanced = pd.concat([df_balanced, df_encoded], axis=1)

    # new list with one hot variables
    list_of_branches_train = list_of_branches_train + list_of_branches_encoded
    if(not is_transform):
        list_of_branches_train.remove(varname)

    return df_balanced, list_of_branches_train, encoder

encoder_njets, encoder_nbjets = None, None
list_of_branches_train_old = list_of_branches_train.copy()
if(do_one_hot_encoding):
    # Encoding m_njets
    df_balanced_train, list_of_branches_train, encoder_njets = do_one_hot_encode_nJets(df_balanced_train, 'm_njets', list_of_branches_train, encoder=None, is_transform=False)
    df_balanced_test, _, _ = do_one_hot_encode_nJets(df_balanced_test, 'm_njets', list_of_branches_train, encoder_njets, True)

    # Encoding m_nbjets
    df_balanced_train, list_of_branches_train, encoder_nbjets = do_one_hot_encode_nJets(df_balanced_train, 'm_nbjets', list_of_branches_train, encoder=None, is_transform=False)
    df_balanced_test, _, _ = do_one_hot_encode_nJets(df_balanced_test, 'm_nbjets', list_of_branches_train, encoder_nbjets, True)

print('------')
print(df_balanced_test.head(10))



# Defining inputs
X_train, X_test = df_balanced_train[list_of_branches_train], df_balanced_test[list_of_branches_train]
Y_train, Y_test = df_balanced_train['class'], df_balanced_test['class']
if is_abs_weight:
    #w_train, w_test = df_balanced_train['weight_nominal_abs'], df_balanced_test['weight_nominal_abs']
    w_train, w_test = df_balanced_train['weight_nominal_abs_corr'], df_balanced_test['weight_nominal_abs_corr']
else:
    w_train, w_test = df_balanced_train['weight_nominal_corr'], df_balanced_test['weight_nominal_corr']
    
# ## Scale Inputs
# X_scaler = StandardScaler() # or MinMaxScaler()
# X_train = X_scaler.fit_transform(X_train)
# X_test = X_scaler.transform(X_test)
# print(pd.DataFrame(X_train, columns=list_of_branches_train, dtype=np.float32))

print('input shape: ', df_balanced_train[list_of_branches_train].shape)
num_features = df_balanced_train[list_of_branches_train].shape[1]
print('number of inputs: ',num_features)
input_shape = (num_features,)

# Model
# Nominal: 'min_child_weight': 1.4, 'max_depth': 6,'eta': 0.17, 'reg_lambda': 6, num_boost_round = 4500
metric = 'auc'
num_boost_round = 2700
parameters = {
    'objective': 'binary:logistic', 
    'booster': 'gbtree', 
    'eval_metric': metric,
    'min_child_weight': 0.2,
    'max_depth': 4,
    'subsample': 0.7,
    #'max_bin' : 10,
    'eta': 0.02, #learning_rate
    'reg_lambda': 0.9
}

searchParams = {
    'learning_rate': [0.02,0.04],
    'n_estimators': [2500, 3500],
    "min_child_weight" : [0.2, 0.4,0.8],
    'max_depth':  [4],
    'reg_lambda' : [0.7, 0.9, 1.5],
    }

#{'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 1500, 'reg_lambda': 1}
#0.7783898656108643
#{'learning_rate': 0.02, 'max_depth': 4, 'min_child_weight': 0.8, 'n_estimators': 2500, 'reg_lambda': 0.5}
#0.7899600217673549
#{'learning_rate': 0.02, 'max_depth': 4, 'min_child_weight': 0.4, 'n_estimators': 2500, 'reg_lambda': 0.7}
#0.8187311999468672
#{'learning_rate': 0.02, 'max_depth': 4, 'min_child_weight': 0.2, 'n_estimators': 3500, 'reg_lambda': 0.9}
#0.8425190696455378
#{'learning_rate': 0.02, 'max_depth': 4, 'min_child_weight': 0.2, 'n_estimators': 2500, 'reg_lambda': 0.9}
#0.8477671889251909 cv=3
#{'learning_rate': 0.02, 'max_depth': 4, 'min_child_weight': 0.2, 'n_estimators': 3500, 'reg_lambda': 0.9}
#0.8562363706610239 cv=4
#{'learning_rate': 0.02, 'max_depth': 4, 'min_child_weight': 0.2, 'n_estimators': 2500, 'reg_lambda': 0.7}
0.8568802278402021
#{'learning_rate': 0.02, 'max_depth': 4, 'min_child_weight': 0.2, 'n_estimators': 2500, 'reg_lambda': 0.9}
#0.865123264023065

train_data = xgb.DMatrix(X_train, label=Y_train, weight=w_train.values, feature_names=list_of_branches_train)
test_data = xgb.DMatrix(X_test, label=Y_test, weight=w_test.values, feature_names=list_of_branches_train)

# Create classifier to use. Note that parameters have to be input manually
# not as a dict!
mXGB = xgb.XGBClassifier(
    objective = 'binary:logistic', 
    booster = 'gbtree', 
    eval_metric = metric,
    min_child_weight = 1.4,
    max_depth = 6,
    subsample = 0.7,
    learning_rate = 0.17,
    n_estimators = 1500,
    reg_lambda = 6)

# Create the grid
# scoring: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
if do_GridSearch:
    grid = GridSearchCV(mXGB, searchParams, cv = 5, scoring="roc_auc", n_jobs=-1, verbose=0) # roc_auc accuracy
    ## Search on whole dataset with cross validation
    #grid.fit(add_events(X_train, X_test), add_events(Y_train, Y_test).values, sample_weight=add_events(w_train, w_test).values)
    grid.fit(X_train, Y_train.values, sample_weight=w_train.values)
    # Print the best parameters found
    print(grid.best_params_)
    print(grid.best_score_)

    # Set the best parameters
    parameters['eta'] = grid.best_params_['learning_rate']
    num_boost_round = grid.best_params_['n_estimators']
    parameters['min_child_weight'] = grid.best_params_['min_child_weight']
    parameters['max_depth'] = grid.best_params_['max_depth']
    parameters['reg_lambda'] = grid.best_params_['reg_lambda']

evals_result = {}  # to record eval results for plotting
model_XGB = xgb.train(parameters,
                       train_data,
                       evals=[(train_data, 'train'), (test_data, 'test')],
                       #early_stopping_rounds=100,
                       evals_result=evals_result,
                       num_boost_round=num_boost_round
                       )

# Plot importance
ax = xgb.plot_importance(model_XGB, max_num_features=20, importance_type='gain') # max_num_features=10 ,  figsize = (12,6)
fig = ax.figure
fig.set_size_inches(12, 6)
plt.savefig('XGB_importance.png')

def plot_metric(evals_result, metric):
    #results = model.evals_result()
    epochs = len(evals_result['test'][metric])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, evals_result['train'][metric], label='Train')
    ax.plot(x_axis, evals_result['test'][metric], label='Test')
    ax.legend()
    plt.ylabel(metric)
    plt.xlabel('Epochs')
    plt.title('XGBoost' + metric)
    plt.savefig('XGB_metric.png')

plot_metric(evals_result, metric=metric)


# ROC curve
y_pred_train = model_XGB.predict(train_data, ntree_limit=model_XGB.best_ntree_limit).ravel()
fpr_train, tpr_train, thresholds_train = roc_curve(Y_train, y_pred_train, sample_weight=w_train.values)
auc_train = auc(fpr_train, tpr_train)

y_pred_test = model_XGB.predict(test_data, ntree_limit=model_XGB.best_ntree_limit).ravel()
fpr_test, tpr_test, thresholds_test = roc_curve(Y_test, y_pred_test, sample_weight=w_test.values)
auc_test = auc(fpr_test, tpr_test)

print(y_pred_train)
print(accuracy_score( np.where(y_pred_test > 0.5, 1, 0), Y_test.values))

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))

plt.figure(3)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_train, tpr_train, label='TF2.0 train (area = {:.3f})'.format(auc_train))
plt.plot(fpr_test, tpr_test, label='TF2.0 test (area = {:.3f})'.format(auc_test))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
#plt.show()
plt.savefig('roc_XGB.png')

# NN-score
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
    plt.xlabel('XGB Score');
    plt.grid()
    #plt.show()
    plt.savefig('score_XGB.png')

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

######################################################
########### Add output to ntuples
######################################################
import glob

ntuplefiles = []
for file in glob.glob("../tH_nominal_1l2tau_"+channel+"_012b_Xj_update/*.root"):
    ntuplefiles.append(file)
#print(ntuplefiles)

if do_update_ntuples:
    for file in ntuplefiles:
        #Get data from each file
        f = ROOT.TFile(file)
        tree = f.Get('tHqLoop_nominal')
        data = rootnp.tree2array(tree, branches=list_of_branches_train_old, 
                        selection="", start=0, stop=None)
        dframe = pd.DataFrame(data, columns=list_of_branches_train_old, dtype=np.float32)
        print(dframe.shape)

        y_pred = None
        if dframe.shape[0]:
            # One Hot Encoding
            if(do_one_hot_encoding):
                dframe, _, _ = do_one_hot_encode_nJets(dframe, 'm_njets', list_of_branches_train, encoder_njets, True)
                dframe, _, _ = do_one_hot_encode_nJets(dframe, 'm_nbjets', list_of_branches_train, encoder_nbjets, True)

            # Getting X
            X = dframe[list_of_branches_train]
            #X = X_scaler.transform(X)
            X_data = xgb.DMatrix(X, label=None, weight=None, feature_names=list_of_branches_train)
            # Prediction
            y_pred = model_XGB.predict(X_data).ravel()
            y_pred.dtype = [('XGBoutput', np.float32)]
            # Update ntuples
            rootnp.array2root(y_pred, file, "tHqLoop_nominal", mode='update')
        else:
            print('The following file is empty: ', file)






