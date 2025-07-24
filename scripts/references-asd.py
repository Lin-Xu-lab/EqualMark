# %%
import pandas as pd
import numpy as np
import os
from itertools import combinations

# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_auc_score

# %%
from sklearn import tree,svm,ensemble

# %%
# %%
def cm_to_metrics(cm):
    acc=(cm[0,0]+cm[1,1])/np.sum(cm)
    if cm[0,1]+cm[1,1]==0:
        precision=np.nan
    else:
        precision=cm[1,1]/(cm[0,1]+cm[1,1])
    if cm[1,0]+cm[1,1]==0:
        recall=np.nan
    else:
        recall=cm[1,1]/(cm[1,0]+cm[1,1])
    if cm[1,0]+cm[0,0]!=0:
        bacc=np.average([precision,cm[0,0]/(cm[1,0]+cm[0,0])])
    else:
        bacc=np.nan
    if precision is not np.nan and recall is not np.nan and precision+recall!=0:
        f1=2*precision*recall/(precision+recall)
    else:
        f1=np.nan
    return acc,precision,recall,f1,bacc

# %%
def stratify_age(x: np.ndarray, middle='median'):
    if middle=='median':
        middle=np.median(x)
    elif middle=='mean':
        middle=np.average(x)
    results=[]
    for i in x:
        if i<=middle:
            results.append("<=%.2f"%middle)
        else:
            results.append(">%.2f"%middle)
    return results, middle

# %%
full_list=['GSE25507_example']
split_dict={
    'GSE25507_example':
    {
        'age',
        'maternal age',
        'paternal age'
    },
}

# %%
append_value_to_dict = lambda dict_name, name1, name2, name3, value : dict_name["-".join([name1,name2,name3])].append(value)

target_col='diagnosis'
criteria='roc_auc'

random_state=5
output_shape=40
num_heads=10
shrinking_speed=2
drop_out=0.1
middle_inflation=8
base_lr=1e-4
shared_output_flag=True
use_gpu=True
use_list=True
middle_inflation_flag=True

# %%
if use_list:
    gene_list_dict={
        'cpn-15-047':
        [
            'HSF2',
            'RFX1',
            'NPM1',
            'MIER2',
            'MRPS31',
            'TC2N',
            'PKM',
            'C12orf29',
            'JADE2',
            'ACKR3',
            'ARHGAP15',
            'ATP6AP1',
            'TAPT1-AS1',
            'TMEM41B',
            'DCAF11'
        ],
    }

# %%
model_dir=os.path.join("models","test")

# %%
criterias=['accuracy','precision','recall','f1','bacc','roc_auc']
results={'split':[],'groups':[],'data_source':[],'dataset_size':[],'biomarker':[],'gene_number':[]}
for md in ['RF','DT','linear_SVM','C_SVM']:
    for cri in criterias:
        results['-'.join(['overall',md,cri])]=[]
        for grp in ['grp0','grp1']:
            results['-'.join([str(grp),md,cri])]=[]

for id in range(1,len(full_list)+1):
    for switch_data_source in combinations(full_list, id):
        for gene_list_name in gene_list_dict.keys():
            gene_list=gene_list_dict[gene_list_name]
            if isinstance(switch_data_source,str):
                df=pd.read_csv(os.path.join("..","data","ASD","joined-%s.tsv"%(switch_data_source)),sep='\t',header=0,index_col=0, low_memory=False)
            else:
                splits=set()
                dfs=[]
                for data in switch_data_source:
                    dfs.append(pd.read_csv(os.path.join("..","data","ASD","joined-%s.tsv"%(data)),sep='\t',header=0,index_col=0, low_memory=False))
                    if splits==set():
                        splits=split_dict[data]
                    else:
                        splits&=split_dict[data]
                df=pd.concat(dfs,axis=0)

            # %%
            for split_col in splits:
                df=pd.concat(dfs,axis=0)
                split_columns=[
                    split_col,
                ]

                # %%
                df=df.loc[df[split_col].notna()&df[target_col].notna(),:]

                # %%
                columns=list(df.columns)

                # %%
                if use_list:
                    used_genes=[]
                    for gene in gene_list:
                        if type(gene)==type("") and gene in df.columns:
                            used_genes.append(gene)
                        elif type(gene)==type([]):
                            for syn_name in gene:
                                if syn_name in df.columns:
                                    used_genes.append(syn_name)
                    gene_list=used_genes
                    print(gene_list)
                    print("using a gene list containing %d genes!"%(len(gene_list)))

                # %%
                if use_list:
                    df=df[gene_list+split_columns+[target_col]]
                else:
                    df=df[columns[:-5]+split_columns+[target_col]]

                # %%
                if 'age' in split_col:
                    df.loc[:,split_col], _ =stratify_age(df[split_col].to_numpy(),middle='median')
                elif 'smoker' in split_col:
                    indices=(df['smoker']=='nonsmoker')
                    df.loc[indices,'smoker']='never'
                    df.loc[[not _ for _ in indices],'smoker']='ever'

                # %%
                to_drop=[]
                for col in gene_list:
                    n_na=np.sum(df[col].isna())
                    if n_na>0:
                        print("%s containing na: %d/%d"%(col,n_na,len(df)))
                        if n_na/len(df)>0.33:
                            print("\tdrop column "+col)
                            to_drop.append(col)
                            df=df.drop(col,axis=1)

                for col in to_drop:
                    gene_list.remove(col)
                print("Now using %d genes!"%(len(gene_list)))

                # %%
                if np.sum(df.isna().values)>0:
                    print("found NaN!")
                    print("original shape: "+str(df.shape))
                    df=df.dropna(axis=0)
                print("shape: "+str(df.shape))

                # %%
                df.loc[df['diagnosis']=='normal','diagnosis']=0
                df.loc[df['diagnosis']=='diabetes','diagnosis']=1
                df['diagnosis']=df['diagnosis'].astype(int)

                # %%
                lengths=[]
                for split in split_columns:
                    for grp in df[split].unique():
                        print("-----------------")
                        print(split+" "+str(grp))
                        print(df.loc[df[split]==grp,target_col].describe())
                        lengths.append(len(df[split]==grp))

                # %% [markdown]
                # # Reference

                # %%
                rf_n_trees=200
                rf_samples_split=6
                rf_min_sample_leaf=15
                dt_samples_split=6
                dt_min_sample_leaf=15
                # nu_SVM_nu=2*min(lengths)/np.sum(lengths)

                # %%
                if len(df)<100 or len(gene_list)==0:
                    print("Dataset too small, skipping...")
                    continue

                # %%
                results['data_source'].append("+".join(switch_data_source))
                results['dataset_size'].append(len(df))
                results['biomarker'].append(gene_list_name)
                results['gene_number'].append(len(gene_list))

                # %%
                for split_i in range(len(split_columns)):
                    results['split'].append(split_columns[split_i])

                    split=split_columns[split_i]

                    print("---------------------")
                    print("using %s as splitting factor:"%split)

                    X_train, X_test, y_train, y_test = train_test_split(df[gene_list+[split]], df['diagnosis'], test_size= 0.2, random_state = random_state)

                    # label encoding is the same as one-hot encoding if there are only two groups
                    labeler=LabelEncoder().fit(X_train[split].values) 
                    print(labeler.classes_)
                    X_train[split]=labeler.transform(X_train[split].values).astype(np.float32)
                    X_test[split]=labeler.transform(X_test[split].values).astype(np.float32)

                    rf=ensemble.RandomForestClassifier(n_estimators=rf_n_trees, min_samples_split=rf_samples_split, min_samples_leaf=rf_min_sample_leaf, n_jobs=-1, random_state=random_state)
                    dt = tree.DecisionTreeClassifier(min_samples_split=dt_samples_split, min_samples_leaf=dt_min_sample_leaf, random_state=random_state)
                    linear_SVM = svm.LinearSVC(dual=True)
                    C_SVM=svm.SVC()
                    # Nu_SVM=svm.NuSVC(nu=nu_SVM_nu)

                    rf.fit(X_train, y_train)
                    dt.fit(X_train,y_train)
                    linear_SVM.fit(X_train, y_train)
                    C_SVM.fit(X_train, y_train)
                    # Nu_SVM.fit(X_train, y_train)

                    rf_predict=rf.predict(X_test)
                    dt_predict=dt.predict(X_test)
                    linear_SVM_predict=linear_SVM.predict(X_test)
                    C_SVM_predict=C_SVM.predict(X_test)
                    # Nu_SVM_predict=Nu_SVM.predict(X_test)

                    cm_rf=confusion_matrix(y_test,rf_predict)
                    cm_dt=confusion_matrix(y_test,dt_predict)
                    cm_linear_SVM=confusion_matrix(y_test,linear_SVM_predict)
                    cm_C_SVM=confusion_matrix(y_test,C_SVM_predict)
                    # cm_Nu_SVM=confusion_matrix(y_test,Nu_SVM_predict)

                    try:
                        roc_auc_rf=roc_auc_score(y_test,rf.predict_proba(X_test)[:,1])
                    except ValueError:
                        roc_auc_rf=np.nan
                    append_value_to_dict(results,'overall','RF','roc_auc',roc_auc_rf)
                    try:
                        roc_auc_dt=roc_auc_score(y_test,dt.predict_proba(X_test)[:,1])
                    except ValueError:
                        roc_auc_dt=np.nan
                    append_value_to_dict(results,'overall','DT','roc_auc',roc_auc_rf)
                    try:
                        roc_auc_linear_SVM=roc_auc_score(y_test,linear_SVM.decision_function(X_test))
                    except ValueError:
                        roc_auc_linear_SVM=np.nan
                    append_value_to_dict(results,'overall','linear_SVM','roc_auc',roc_auc_rf)
                    try:
                        roc_auc_C_SVM=roc_auc_score(y_test,C_SVM.decision_function(X_test))
                    except ValueError:
                        roc_auc_C_SVM=np.nan
                    append_value_to_dict(results,'overall','C_SVM','roc_auc',roc_auc_rf)

                    if cm_rf.shape==(2,2):
                        values = cm_to_metrics(cm_rf)
                        print("RF performances: \tAccuracy %f, \tPrecision %f, \tRecall %f, \tF1 score %f, \tBalanced accuracy %f"%(values),end='')
                        print(", \tROC-AUC score %f"%(roc_auc_rf))
                        for i in range(len(['accuracy','precision','recall','f1','bacc'])):
                            append_value_to_dict(results,'overall','RF',criterias[i],values[i])
                    else:
                        values = (np.nan,np.nan,np.nan,np.nan,np.nan)
                        print("unable to calculate metrics due to lack of classes")
                        for i in range(len(['accuracy','precision','recall','f1','bacc'])):
                            append_value_to_dict(results,'overall','RF',criterias[i],values[i])
                    if cm_dt.shape==(2,2):
                        values = cm_to_metrics(cm_dt)
                        print("DT performances: \tAccuracy %f, \tPrecision %f, \tRecall %f, \tF1 score %f, \tBalanced accuracy %f"%(values),end='')
                        print(", \tROC-AUC score %f"%(roc_auc_dt))
                        for i in range(len(['accuracy','precision','recall','f1','bacc'])):
                            append_value_to_dict(results,'overall','DT',criterias[i],values[i])
                    else:
                        values = (np.nan,np.nan,np.nan,np.nan,np.nan)
                        print("unable to calculate metrics due to lack of classes")
                        for i in range(len(['accuracy','precision','recall','f1','bacc'])):
                            append_value_to_dict(results,'overall','DT',criterias[i],values[i])
                    if cm_linear_SVM.shape==(2,2):
                        values = cm_to_metrics(cm_linear_SVM)
                        print("linear_SVM performances: \tAccuracy %f, \tPrecision %f, \tRecall %f, \tF1 score %f, \tBalanced accuracy %f"%(values),end='')
                        print(", \tROC-AUC score %f"%(roc_auc_linear_SVM))
                        for i in range(len(['accuracy','precision','recall','f1','bacc'])):
                            append_value_to_dict(results,'overall','linear_SVM',criterias[i],values[i])
                    else:
                        values = (np.nan,np.nan,np.nan,np.nan,np.nan)
                        print("unable to calculate metrics due to lack of classes")
                        for i in range(len(['accuracy','precision','recall','f1','bacc'])):
                            append_value_to_dict(results,'overall','linear_SVM',criterias[i],values[i])
                    if cm_C_SVM.shape==(2,2):
                        values = cm_to_metrics(cm_C_SVM)
                        print("C_SVM performances: \tAccuracy %f, \tPrecision %f, \tRecall %f, \tF1 score %f, \tBalanced accuracy %f"%(values),end='')
                        print(", \tROC-AUC score %f"%(roc_auc_C_SVM))
                        for i in range(len(['accuracy','precision','recall','f1','bacc'])):
                            append_value_to_dict(results,'overall','C_SVM',criterias[i],values[i])
                    else:
                        values = (np.nan,np.nan,np.nan,np.nan,np.nan)
                        print("unable to calculate metrics due to lack of classes")
                        for i in range(len(['accuracy','precision','recall','f1','bacc'])):
                            append_value_to_dict(results,'overall','C_SVM',criterias[i],values[i])
                    # if cm_Nu_SVM.shape==(2,2):
                    #     print("Nu support SVM performances: \tAccuracy %f, \tPrecision %f, \tRecall %f, \tF1 score %f, \tBalanced accuracy %f"%cm_to_metrics(cm_Nu_SVM))
                    # else:
                    #     print("unable to calculate metrics due to lack of classes")

                    print()
                    groups=list(X_test[split].unique())
                    tmp={}
                    for grp in groups:
                        tmp[labeler.classes_[int(grp)]]=np.sum(X_test[split]==grp)
                    results['groups'].append(str(tmp))
                    print("group-wise results:")
                    for ii in range(len(groups)):
                        group=groups[ii]
                        indices=X_test[X_test[split]==group].index
                        print("%s group contain %d patients"%("grp%d"%ii, len(indices)))
                        group_test=X_test.loc[indices].copy()
                        group_test_y=y_test[[X_test.index.get_loc(_) for _ in indices]].copy()
                        rf_predict=rf.predict(group_test)
                        dt_predict=dt.predict(group_test)
                        linear_SVM_predict=linear_SVM.predict(group_test)
                        C_SVM_predict=C_SVM.predict(group_test)
                        # Nu_SVM_predict=Nu_SVM.predict(group_test)

                        cm_rf=confusion_matrix(group_test_y,rf_predict)
                        cm_dt=confusion_matrix(group_test_y,dt_predict)
                        cm_linear_SVM=confusion_matrix(group_test_y,linear_SVM_predict)
                        cm_C_SVM=confusion_matrix(group_test_y,C_SVM_predict)
                        # cm_Nu_SVM=confusion_matrix(group_test_y,Nu_SVM_predict)

                        try:
                            roc_auc_rf=roc_auc_score(group_test_y,rf.predict_proba(group_test)[:,1])
                        except ValueError:
                            roc_auc_rf=np.nan
                        append_value_to_dict(results,"grp%d"%ii,'RF','roc_auc',roc_auc_rf)
                        try:
                            roc_auc_dt=roc_auc_score(group_test_y,dt.predict_proba(group_test)[:,1])
                        except ValueError:
                            roc_auc_dt=np.nan
                        append_value_to_dict(results,"grp%d"%ii,'DT','roc_auc',roc_auc_dt)
                        try:
                            roc_auc_linear_SVM=roc_auc_score(group_test_y,linear_SVM.decision_function(group_test))
                        except ValueError:
                            roc_auc_linear_SVM=np.nan
                        append_value_to_dict(results,"grp%d"%ii,'linear_SVM','roc_auc',roc_auc_linear_SVM)
                        try:
                            roc_auc_C_SVM=roc_auc_score(group_test_y,C_SVM.decision_function(group_test))
                        except ValueError:
                            roc_auc_C_SVM=np.nan
                        append_value_to_dict(results,"grp%d"%ii,'C_SVM','roc_auc',roc_auc_C_SVM)

                        if cm_rf.shape==(2,2):
                            values = cm_to_metrics(cm_rf)
                            print("RF performances: \tAccuracy %f, \tPrecision %f, \tRecall %f, \tF1 score %f, \tBalanced accuracy %f"%(values),end='')
                            print(", \tROC-AUC score %f"%(roc_auc_rf))
                        else:
                            values = (np.nan,np.nan,np.nan,np.nan,np.nan)
                            print("unable to calculate metrics due to lack of classes")
                        for i in range(len(['accuracy','precision','recall','f1','bacc'])):
                            append_value_to_dict(results,"grp%d"%ii,'RF',criterias[i],values[i])

                        if cm_dt.shape==(2,2):
                            values = cm_to_metrics(cm_dt)
                            print("DT performances: \tAccuracy %f, \tPrecision %f, \tRecall %f, \tF1 score %f, \tBalanced accuracy %f"%(values),end='')
                            print(", \tROC-AUC score %f"%(roc_auc_dt))
                        else:
                            values = (np.nan,np.nan,np.nan,np.nan,np.nan)
                            print("unable to calculate metrics due to lack of classes")
                        for i in range(len(['accuracy','precision','recall','f1','bacc'])):
                            append_value_to_dict(results,"grp%d"%ii,'DT',criterias[i],values[i])

                        if cm_linear_SVM.shape==(2,2):
                            values = cm_to_metrics(cm_linear_SVM)
                            print("linear_SVM performances: \tAccuracy %f, \tPrecision %f, \tRecall %f, \tF1 score %f, \tBalanced accuracy %f"%(values),end='')
                            print(", \tROC-AUC score %f"%(roc_auc_linear_SVM))
                        else:
                            values = (np.nan,np.nan,np.nan,np.nan,np.nan)
                            print("unable to calculate metrics due to lack of classes")
                        for i in range(len(['accuracy','precision','recall','f1','bacc'])):
                            append_value_to_dict(results,"grp%d"%ii,'linear_SVM',criterias[i],values[i])

                        if cm_C_SVM.shape==(2,2):
                            values = cm_to_metrics(cm_C_SVM)
                            print("C_SVM performances: \tAccuracy %f, \tPrecision %f, \tRecall %f, \tF1 score %f, \tBalanced accuracy %f"%(values),end='')
                            print(", \tROC-AUC score %f"%(roc_auc_C_SVM))
                        else:
                            values = (np.nan,np.nan,np.nan,np.nan,np.nan)
                            print("unable to calculate metrics due to lack of classes")
                        for i in range(len(['accuracy','precision','recall','f1','bacc'])):
                            append_value_to_dict(results,"grp%d"%ii,'C_SVM',criterias[i],values[i])

                        # if cm_Nu_SVM.shape==(2,2):
                        #     print("Nu support SVM performances: \tAccuracy %f, \tPrecision %f, \tRecall %f, \tF1 score %f, \tBalanced accuracy %f"%cm_to_metrics(cm_Nu_SVM))
                        # else:
                        #     print("unable to calculate metrics due to lack of classes")

                        print()

# %%
pd.DataFrame(results).to_csv("references-asd.tsv",sep='\t',index=True,header=True)
