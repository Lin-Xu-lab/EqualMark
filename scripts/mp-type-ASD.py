# %%
from torch import multiprocessing as mp
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

# %%
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# %%
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# %%
class TypeData(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# %%
def compute_layer_sizes(input_shape:int, output_shape:int=1, shrinking_speed=4):
    li=[]
    while(input_shape>output_shape):
        x=input_shape//shrinking_speed
        li.append(x)
        input_shape=x
    if len(li)>0 and li[-1]<=output_shape:
            li.remove(li[-1])
    return li

# %%
class Expert(nn.Module):
    def __init__(self, input_shape, output_shape, shrinking_speed=4, drop_out=0.3, middle_inflation_flag=False, middle_inflation=2, classification_flag=False) -> None:
        super().__init__()
        self.layers=nn.Sequential()
        if middle_inflation_flag:
            inflated_middle=int(input_shape*middle_inflation)
            layer_sizes1=compute_layer_sizes(inflated_middle, input_shape, shrinking_speed)[::-1]
            for layer_n in range(len(layer_sizes1)):
                if layer_n==0:
                    pre=input_shape
                else:
                    pre=layer_sizes1[layer_n-1]

                post=layer_sizes1[layer_n]
                self.layers.append(nn.Linear(pre, post))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(drop_out))
                self.layers.append(nn.BatchNorm1d(post))
            
            if len(layer_sizes1)==0:
                self.layers.append(nn.Linear(input_shape,inflated_middle))
            else:
                self.layers.append(nn.Linear(post,inflated_middle))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(drop_out))
                self.layers.append(nn.BatchNorm1d(inflated_middle))
            layer_sizes2=compute_layer_sizes(inflated_middle, output_shape, shrinking_speed)
            for layer_n in range(len(layer_sizes2)):
                if layer_n==0:
                    pre=inflated_middle
                else:
                    pre=layer_sizes2[layer_n-1]

                post=layer_sizes2[layer_n]
                self.layers.append(nn.Linear(pre, post))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(drop_out))
                self.layers.append(nn.BatchNorm1d(post))
            
            if len(layer_sizes2)==0:
                self.layers.append(nn.Linear(inflated_middle,output_shape))
            else:
                self.layers.append(nn.Linear(post,output_shape))
        else:
            layer_sizes=compute_layer_sizes(input_shape, output_shape, shrinking_speed)
            for layer_n in range(len(layer_sizes)):
                if layer_n==0:
                    pre=input_shape
                else:
                    pre=layer_sizes[layer_n-1]

                post=layer_sizes[layer_n]
                self.layers.append(nn.Linear(pre, post))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(drop_out))
                self.layers.append(nn.BatchNorm1d(post))
            
            if len(layer_sizes)==0:
                self.layers.append(nn.Linear(input_shape,output_shape))
            else:
                self.layers.append(nn.Linear(post,output_shape))

        if classification_flag:
            self.layers.append(nn.Softmax(dim=1))
        self.output_features=output_shape

    def forward(self, inputs):
        return self.layers(inputs)

# %%
def confusion_matrix(labels,preds):
    assert len(labels)==len(preds), "labels and preds have different lengths!"
    cm=np.zeros((2,2))
    for i in range(len(labels)):
        if labels[i]==0 and preds[i]==0:
            # true negative
            cm[0,0]+=1
        elif labels[i]==1 and preds[i]==0:
            # false negative
            cm[1,0]+=1
        elif labels[i]==0 and preds[i]==1:
            # false positive
            cm[0,1]+=1
        elif labels[i]==1 and preds[i]==1:
            # true positive
            cm[1,1]+=1
    assert np.sum(cm)==len(labels), "cm calculation error!"
    return cm

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
    if cm[1,0]+cm[0,0]!=0 and precision is not np.nan:
        bacc=np.average([precision,cm[0,0]/(cm[1,0]+cm[0,0])])
    else:
        bacc=np.nan
    if precision is not np.nan and recall is not np.nan and precision+recall!=0:
        f1=2*precision*recall/(precision+recall)
    else:
        f1=np.nan
    return acc,precision,recall,f1,bacc

# %%
def train_type(dataloader, model, loss_fn, optimizer, device, verbose=1):
    model.train()
    model.to(device)
    for batch, (X, Y) in enumerate(dataloader):
        X=X.to(device)
        Y = Y.to(device)

        pred = model(X)
        loss = loss_fn(pred, Y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred_logits = pred.detach().to('cpu')
        cm=confusion_matrix(np.argmax(Y.to('cpu').numpy(),axis=1),np.argmax(pred_logits.numpy(),axis=1))
        try:
            acc,precision,recall,f1,bacc=cm_to_metrics(cm)
        except IndexError:
            if all(np.argmax(pred_logits,axis=1)):
                acc,precision,recall,f1,bacc=1,1,1,1,1
            else:
                acc,precision,recall,f1,bacc=1,np.nan,np.nan,np.nan,np.nan

        try:
            roc_auc=roc_auc_score(Y.to('cpu'),pred_logits)
        except ValueError:
            roc_auc=np.nan

        if batch % 1 == 0:
            loss, _ = loss.item(), (batch + 1) * X.shape[0]
            if verbose>0:
                print(f"training: \tloss {loss:>7f},\t Confusion matrix {cm},\t accuracy {acc},\t precision {precision},\t recall {recall},\t f1 {f1},\t balanced accuracy {bacc},\t ROC AUC {roc_auc}")

# %%
def test_type(dataloader, model: nn.Module, loss_fn, phase: str, device, verbose=0):
    num_batches = len(dataloader)
    model.eval()
    model.to(device)
    test_loss = 0
    with torch.no_grad():
        for X, Y in dataloader:
            X= X.to(device)
            Y= Y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, Y).item()

            pred_logits = pred.to('cpu')
            cm=confusion_matrix(np.argmax(Y.to('cpu').numpy(),axis=1),np.argmax(pred_logits.numpy(),axis=1))
            try:
                acc,precision,recall,f1,bacc=cm_to_metrics(cm)
            except IndexError:
                if all(np.argmax(pred_logits,axis=1)):
                    acc,precision,recall,f1,bacc=1,1,1,1,1
                else:
                    acc,precision,recall,f1,bacc=1,np.nan,np.nan,np.nan,np.nan
            try:
                roc_auc=roc_auc_score(Y.to('cpu'),pred.to("cpu"))
            except ValueError:
                roc_auc=np.nan

    test_loss /= num_batches
    if verbose>0:
        print(f"training: \tloss {test_loss:>7f},\t Confusion matrix {cm},\t accuracy {acc},\t precision {precision},\t recall {recall},\t f1 {f1},\t balanced accuracy {bacc}, \t ROC AUC {roc_auc}")
    return test_loss, cm, (acc, precision, recall, f1, bacc, roc_auc), (Y.to('cpu').numpy(), pred_logits.numpy())

# %%
# Define classification model
class TypeAttnNet(nn.Module):
    def __init__(self, input_shape, shared_expert: nn.Module, share_lin_attn: nn.Module, share_attention: nn.Module, conf_experts: nn.ModuleList, shared_output_flag: bool, num_classes: int, output_layers: nn.Module = None, num_heads :int = 1, shrinking_speed=4, drop_out=0.3, middle_inflation_flag=False, middle_inflation=2):
        super().__init__()
        self.shared_output_flag=shared_output_flag
        self.shared_expert=shared_expert
        self.conf_lin_attn=nn.ModuleList()
        self.conf_attentions=nn.ModuleList()
        self.conf_experts=conf_experts
        for i in range(len(self.conf_experts)):
            self.conf_lin_attn.append(nn.Sequential())
            conf_expert_output=self.conf_experts[i].output_features
            assert conf_expert_output%num_heads==0, "Invalid value for num_heads! It must be dividable for conf_expert_output, which is %d!"%(conf_expert_output)
            self.conf_lin_attn[i].append(Expert(input_shape, conf_expert_output, shrinking_speed=shrinking_speed, drop_out=drop_out, middle_inflation_flag=middle_inflation_flag, middle_inflation=middle_inflation))
            self.conf_attentions.append(nn.MultiheadAttention(1,1,dropout=drop_out,batch_first=True))
        self.share_lin_attn=share_lin_attn
        self.share_attention=share_attention
        if shared_output_flag:
            self.shared_output_layers=output_layers
        else:
            self.output_layers=Expert(self.shared_expert.output_features+sum([x.output_features for x in self.conf_experts]), num_classes, shrinking_speed=shrinking_speed, drop_out=drop_out, classification_flag=True)

    def forward(self, inputs):
        experts=[]
        e_shared=self.shared_expert(inputs)
        e_shared=e_shared.reshape(e_shared.shape[0],e_shared.shape[1],1)
        e_share_lin_attn=self.share_lin_attn(inputs)
        e_share_lin_attn=e_share_lin_attn.reshape(e_share_lin_attn.shape[0],e_share_lin_attn.shape[1],1)
        e_share_attn, _=self.share_attention(e_share_lin_attn, e_shared, e_shared)
        experts.append(e_share_attn.reshape(e_share_attn.shape[0],e_share_attn.shape[1]))
        for i in range(len(self.conf_experts)):
            e_conf=self.conf_experts[i](inputs)
            e_conf=e_conf.reshape(e_conf.shape[0],e_conf.shape[1],1)
            e_conf_lin_attn=self.conf_lin_attn[i](inputs)
            e_conf_lin_attn=e_conf_lin_attn.reshape(e_conf_lin_attn.shape[0],e_conf_lin_attn.shape[1],1)
            e_conf_attn, _=self.conf_attentions[i](e_conf_lin_attn, e_conf, e_conf)
            e_conf_attn=e_conf_attn.reshape(e_conf_attn.shape[0],e_conf_attn.shape[1])
            experts.append(e_conf_attn)
        x = torch.cat(experts,dim=1)
        if self.shared_output_flag:
            outputs=self.shared_output_layers(x)
        else:
            outputs=self.output_layers(x)
        return outputs

# %%
def collector_single(dir_name,result_queue: mp.Queue, length):
    results = []
    pbar = tqdm(range(length),desc="single mode")
    pbar.update(len(results))
    pbar.refresh()
    while True:
        result = result_queue.get()
        if result == "STOP":
            pbar.close()
            break
        results.append(pd.DataFrame(result))
        pbar.update()
        pbar.refresh()

    results_df=pd.concat(results)
    results_df.to_csv(os.path.join(dir_name,"results_single.tsv"),index=False,sep='\t')

# %%
def collector_multi(dir_name,result_queue: mp.Queue, length):
    results = []
    pbar = tqdm(range(length),desc="multi mode")
    pbar.update(len(results))
    pbar.refresh()
    while True:
        result = result_queue.get()
        if result == "STOP":
            pbar.close()
            break
        results.append(pd.DataFrame(result))
        pbar.update()
        pbar.refresh()

    results_df=pd.concat(results)
    results_df.to_csv(os.path.join(dir_name,"results.tsv"),index=False,sep='\t')

# %%
def worker_single(model_id: int, FLAGS, device, middle_inflation_flag, base_lr, single_parameters, gene_list, split_columns, columns, df: pd.DataFrame, queue: mp.Queue):
    results_dict={"model_id":[],"random_state":[], "shrinking_speed":[], "drop_out":[],"middle_inflation":[],"best_overall_"+FLAGS.criteria:[]}
    loss_fn=nn.CrossEntropyLoss()

    results_dict['model_id'].append(model_id)
    random_state=single_parameters[model_id][0]
    shrinking_speed=single_parameters[model_id][1]
    drop_out=single_parameters[model_id][2]
    middle_inflation=single_parameters[model_id][3]

    # %%
    X=df[gene_list+split_columns].copy()
    Y=df[FLAGS.target_col].to_numpy().astype(np.float32)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    # %%
    # scaler=MinMaxScaler().fit(X_train.iloc[:,:-1].to_numpy())
    # X_train.iloc[:,:-1]=scaler.transform(X_train.iloc[:,:-1].to_numpy()).astype(np.float32)
    # X_test.iloc[:,:-1]=scaler.transform(X_test.iloc[:,:-1].to_numpy()).astype(np.float32)

    labeler=LabelEncoder().fit(X_train.iloc[:,-1].to_numpy())
    for cri in ['accuracy', 'precision','recall','f1','bacc','roc_auc']:
        results_dict['overall_'+cri]=[]
        results_dict['delta_'+cri]=[]
        results_dict['relative_delta_'+cri]=[]
        for lab in labeler.classes_:
            if 'best_'+lab+'_'+cri not in results_dict.keys():
                results_dict['best_'+lab+'_'+cri]=[]
                results_dict['best_overall_'+lab+'_'+cri]=[]
    X_train.iloc[:,-1]=labeler.transform(X_train.iloc[:,-1].to_numpy()).astype(np.float32)
    X_test.iloc[:,-1]=labeler.transform(X_test.iloc[:,-1].to_numpy()).astype(np.float32)

    X_train=X_train.to_numpy().astype(np.float32)
    X_test=X_test.to_numpy().astype(np.float32)

    oel=OneHotEncoder(dtype=np.float32).fit(Y_train.reshape(-1,1))
    Y_train=oel.transform(Y_train.reshape(-1,1)).toarray()
    Y_test=oel.transform(Y_test.reshape(-1,1)).toarray()

    # %%
    train_dataset=TypeData(X_train, Y_train)
    train_dataloader=DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataset=TypeData(X_test, Y_test)
    test_dataloader=DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # %%
    results_dict['random_state'].append(random_state)
    results_dict['shrinking_speed'].append(shrinking_speed)
    results_dict['drop_out'].append(drop_out)
    results_dict['middle_inflation'].append(middle_inflation)

    ex=Expert(X_train.shape[1],Y_train.shape[-1],shrinking_speed=shrinking_speed,drop_out=drop_out, middle_inflation_flag=middle_inflation_flag, middle_inflation=middle_inflation, classification_flag=True).to(device)
    optim=torch.optim.Adam(ex.parameters(),lr=base_lr)

    # %%
    #training neural network
    key_list=list(labeler.classes_)
    best_cri=0
    best_grp_cri={}
    overall_cri={}
    for key in key_list:
        best_grp_cri[key]={'accuracy':0, 'precision':0, 'recall':0, 'f1':0, 'bacc':0,'roc_auc':0}
        overall_cri[key]={'accuracy':0, 'precision':0, 'recall':0, 'f1':0, 'bacc':0,'roc_auc':0}
    overall={'accuracy':0, 'precision':0, 'recall':0, 'f1':0, 'bacc':0,'roc_auc':0}
    epochs = 500

    for t in range(epochs):
        train_type(train_dataloader, ex, loss_fn, optim, device, verbose=0)

        test_loss, test_cm, (test_acc, test_precision, test_recall, test_f1, test_bacc, test_roc_auc), (test_y_np, test_pred_np)= test_type(test_dataloader, ex, loss_fn, "test", device)
        if FLAGS.criteria=='accuracy':
            test_cri=test_acc
        elif FLAGS.criteria=='precision':
            test_cri=test_precision
        elif FLAGS.criteria=='recall':
            test_cri=test_recall
        elif FLAGS.criteria=='f1':
            test_cri=test_f1
        elif FLAGS.criteria=='bacc':
            test_cri=test_bacc
        elif FLAGS.criteria=='roc_auc':
            test_cri=test_roc_auc
        
        temp_grp_cri={}
        for i in range(len(key_list)):
            grp_ind=X_test[:,-1]==i
            grp_dataset=TypeData(X_test[grp_ind], Y_test[grp_ind])
            tmp_loss, tmp_cm, (tmp_acc, tmp_precision, tmp_recall, tmp_f1, tmp_bacc, tmp_roc_auc), (tmp_y_np, tmp_pred_np)= test_type(DataLoader(grp_dataset, batch_size=len(grp_dataset), shuffle=False), ex, loss_fn, 'test', device)
            temp_grp_cri[key_list[i]]={}
            temp_grp_cri[key_list[i]]['accuracy']=tmp_acc
            temp_grp_cri[key_list[i]]['precision']=tmp_precision
            temp_grp_cri[key_list[i]]['recall']=tmp_recall
            temp_grp_cri[key_list[i]]['f1']=tmp_f1
            temp_grp_cri[key_list[i]]['bacc']=tmp_bacc
            temp_grp_cri[key_list[i]]['roc_auc']=tmp_roc_auc
        
        if test_cri>best_cri:
            best_cri=test_cri
            for key in temp_grp_cri.keys():
                overall_cri[key]=temp_grp_cri[key]
                overall['accuracy']=test_acc
                overall['precision']=test_precision
                overall['recall']=test_recall
                overall['f1']=test_f1
                overall['bacc']=test_bacc
                overall['roc_auc']=test_roc_auc

        for key in key_list:
            for cri in ['accuracy', 'precision','recall','f1','bacc','roc_auc']:
                if temp_grp_cri[key][cri]>best_grp_cri[key][cri]:
                    best_grp_cri[key][cri]=temp_grp_cri[key][cri]


    # %%
    results_dict['best_overall_'+FLAGS.criteria].append(best_cri)
    for cri in ['accuracy', 'precision','recall','f1','bacc','roc_auc']:
        results_dict['overall_'+cri].append(overall[cri])
        tmp=[]
        for key in key_list:
            results_dict['best_'+key+'_'+cri].append(best_grp_cri[key][cri])
            results_dict['best_overall_'+key+'_'+cri].append(overall_cri[key][cri])
            tmp.append(overall_cri[key][cri])
        results_dict['delta_'+cri].append(max(tmp)-min(tmp))
        if results_dict['overall_'+cri][-1]!=0 and results_dict['overall_'+cri][-1] is not np.nan:
            results_dict['relative_delta_'+cri].append(results_dict['delta_'+cri][-1]/results_dict['overall_'+cri][-1])
        else:
            results_dict['relative_delta_'+cri].append(np.nan)

    queue.put(results_dict)

# %%
def worker_multi(para_id, FLAGS, device, multi_parameters, dataset_dict, model_dir, minimum_class, gene_list, columns, split_columns, key_list, middle_inflation_flag, num_heads, base_lr, result_queue: mp.Queue):
    loss_fn=nn.CrossEntropyLoss()
    random_state,output_shape,drop_out,shrinking_speed,middle_inflation = multi_parameters[para_id]

    results_dict={"model_id":[],"random_state":[], "output_shape":[], "shrinking_speed":[], "drop_out":[], "middle_inflation":[], "best_overall_"+FLAGS.criteria:[], "supp_overall_"+FLAGS.criteria:[]}
    results_dict['model_id'].append(para_id)
    for cri in ['accuracy', 'precision','recall','f1','bacc','roc_auc']:
        results_dict['overall_'+cri]=[]
        results_dict['supp_'+cri]=[]
        results_dict['delta_'+cri]=[]
        results_dict['relative_delta_'+cri]=[]
        results_dict['supp_delta_'+cri]=[]
        results_dict['supp_relative_delta_'+cri]=[]
        for key in key_list:
            results_dict['best_'+key+'_'+cri]=[]
            results_dict['best_overall_'+key+'_'+cri]=[]
            results_dict['supp_'+key+'_'+cri]=[]

    for key, dic in dataset_dict.items():
        X=dic['df'][gene_list].copy().to_numpy().astype(np.float32)
        Y=dic['df'][FLAGS.target_col].to_numpy().astype(np.float32)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

        oel=OneHotEncoder(dtype=np.float32).fit(Y_train.reshape(-1,1))
        Y_train=oel.transform(Y_train.reshape(-1,1)).toarray()
        Y_test=oel.transform(Y_test.reshape(-1,1)).toarray()

        dic['X_train']=X_train
        dic['X_test']=X_test
        dic['Y_train']=Y_train
        dic['Y_test']=Y_test

    # %%
    input_shape=len(gene_list)

    # %%
    results_dict['random_state'].append(random_state)
    results_dict['output_shape'].append(output_shape)
    results_dict['shrinking_speed'].append(shrinking_speed)
    results_dict['drop_out'].append(drop_out)
    results_dict['middle_inflation'].append(middle_inflation)

    for key in key_list:
        dataset_dict[key]['train_dataset']=TypeData(dataset_dict[key]['X_train'], dataset_dict[key]['Y_train'])
        dataset_dict[key]['train_dataloader']=DataLoader(dataset_dict[key]['train_dataset'], batch_size=len(dataset_dict[key]['X_train']))
        dataset_dict[key]['test_dataset']=TypeData(dataset_dict[key]['X_test'], dataset_dict[key]['Y_test'])
        dataset_dict[key]['test_dataloader']=DataLoader(dataset_dict[key]['test_dataset'], batch_size=len(dataset_dict[key]['X_test']))
        dataset_dict[key]['conf_expert']=Expert(input_shape=input_shape, output_shape=output_shape, drop_out=drop_out, shrinking_speed=shrinking_speed, middle_inflation_flag=middle_inflation_flag, middle_inflation=middle_inflation)

    # %%
    shared_expert=Expert(input_shape=input_shape, output_shape=output_shape, drop_out=drop_out, shrinking_speed=shrinking_speed, middle_inflation_flag=middle_inflation_flag, middle_inflation=middle_inflation)
    share_lin_attn=Expert(input_shape, shared_expert.output_features, shrinking_speed=shrinking_speed, drop_out=drop_out, middle_inflation_flag=middle_inflation_flag, middle_inflation=middle_inflation)
    share_attention=nn.MultiheadAttention(1,1, dropout=drop_out, batch_first=True)
    if FLAGS.shared_output_flag:
        output_layers=Expert(shared_expert.output_features+len(split_columns)*output_shape, Y_train.shape[-1], shrinking_speed=shrinking_speed, drop_out=drop_out, middle_inflation_flag=middle_inflation_flag, middle_inflation=middle_inflation, classification_flag=True)
        shared_expert_optim=torch.optim.Adam([
            {'params': shared_expert.parameters()},
            {'params': share_lin_attn.parameters()},
            {'params': share_attention.parameters()},
            {'params': output_layers.parameters()}
        ],lr=base_lr,weight_decay=16)
    else:
        shared_expert_optim=torch.optim.Adam([
            {'params': shared_expert.parameters()},
            {'params': share_lin_attn.parameters()},
            {'params': share_attention.parameters()}
        ],lr=base_lr,weight_decay=16)

    # %%
    for key in key_list:
        if FLAGS.shared_output_flag:
            dataset_dict[key]['model']=TypeAttnNet(input_shape=input_shape, shared_expert= shared_expert, share_lin_attn= share_lin_attn, share_attention= share_attention, conf_experts= nn.ModuleList([dataset_dict[key]['conf_expert']]), shared_output_flag=FLAGS.shared_output_flag, num_classes=Y_train.shape[-1], output_layers=output_layers, num_heads=num_heads, drop_out=drop_out, shrinking_speed=shrinking_speed)
        else:
            dataset_dict[key]['model']=TypeAttnNet(input_shape=input_shape, shared_expert= shared_expert, share_lin_attn= share_lin_attn, share_attention= share_attention, conf_experts= nn.ModuleList([dataset_dict[key]['conf_expert']]), shared_output_flag=FLAGS.shared_output_flag, num_classes=Y_train.shape[-1], num_heads=num_heads, drop_out=drop_out, shrinking_speed=shrinking_speed)
        shared_expert_params = [p for name, p in dataset_dict[key]['model'].named_parameters() if 'share' in name]
        others = [p for name, p in dataset_dict[key]['model'].named_parameters() if 'share' not in name]
        dataset_dict[key]['optimizer']=torch.optim.Adam([
            {'params':others},
            {'params':shared_expert_params, 'lr':0}
        ],lr=base_lr*(minimum_class/len(dataset_dict[key]['df'])),weight_decay=16)

    # %%
    #training neural network
    best_overall_cri=0
    overall_grp_cri={}
    best_grp_cri={}
    for key in key_list:
        best_grp_cri[key]={'accuracy':0, 'precision':0, 'recall':0, 'f1':0, 'bacc':0, 'roc_auc':0}
        overall_grp_cri[key]={'accuracy':0, 'precision':0, 'recall':0, 'f1':0, 'bacc':0, 'roc_auc':0}
    epochs = 500
    saved=False
    overall={'accuracy':0, 'precision':0, 'recall':0, 'f1':0, 'bacc':0,'roc_auc':0}
    for t in range(epochs):
        shared_expert.train()
        shared_expert.to(device)
        for i in range(len(key_list)):
            key=key_list[i]
            dataloader_train_surv = dataset_dict[key]['train_dataloader']
            surv_model = dataset_dict[key]['model']

            surv_model.train()
            surv_model.to(device)
            for batch, (X, Y) in enumerate(dataloader_train_surv):
                X=X.to(device)
                Y=Y.to(device)

                pred = surv_model(X)
                temp_loss = loss_fn(pred, Y)
            
            if i==0:
                loss=temp_loss
            else:
                loss+=temp_loss

        loss.backward()
        shared_expert_optim.step()
        for key in key_list:
            opt = dataset_dict[key]['optimizer']
            opt.step()
        shared_expert_optim.zero_grad()
        for key in key_list:
            opt = dataset_dict[key]['optimizer']
            opt.zero_grad()

        for key in key_list:
            # log_fi.write(key+" training set: ")
            dataloader_train_surv = dataset_dict[key]['train_dataloader']
            model = dataset_dict[key]['model']
            key_loss, key_cm, (key_acc, key_precision, key_recall, key_f1, key_bacc, key_roc_auc), (key_y_np, key_pred_np) = test_type(dataloader_train_surv, model, loss_fn, "train", device)

        test_grp_cri={}
        cms=np.zeros(key_cm.shape)
        for key in key_list:
            test_grp_cri[key]={}
            dataloader_test_surv = dataset_dict[key]['test_dataloader']
            model = dataset_dict[key]['model']
            tmp_loss, tmp_cm, (tmp_acc, tmp_precision, tmp_recall, tmp_f1, tmp_bacc, tmp_roc_auc), (tmp_y_np, tmp_pred_np) = test_type(dataloader_test_surv, model, loss_fn, "test", device)
            test_grp_cri[key]['accuracy']=tmp_acc
            test_grp_cri[key]['precision']=tmp_precision
            test_grp_cri[key]['recall']=tmp_recall
            test_grp_cri[key]['f1']=tmp_f1
            test_grp_cri[key]['bacc']=tmp_bacc
            test_grp_cri[key]['roc_auc']=tmp_roc_auc
            test_grp_cri[key]['y_np']=tmp_y_np
            test_grp_cri[key]['pred_np']=tmp_pred_np
            cms+=tmp_cm
        
        try:
            overall_acc, overall_precision, overall_recall, overall_f1, overall_bacc=cm_to_metrics(cms)
        except IndexError:
            if cms[1,1]==0:
                overall_acc, overall_precision, overall_recall, overall_f1, overall_bacc=1,np.nan,np.nan,np.nan,np.nan
            else:
                overall_acc, overall_precision, overall_recall, overall_f1, overall_bacc=1,1,1,1,np.nan
        try:
            overall_y_np=np.concatenate([test_grp_cri[key]['y_np'] for key in key_list])
            overall_pred_np=np.concatenate([test_grp_cri[key]['pred_np'] for key in key_list])
            overall_roc_auc=roc_auc_score(overall_y_np,overall_pred_np)
        except ValueError:
            overall_roc_auc=np.nan

        if FLAGS.criteria=='accuracy':
            test_cri=overall_acc
        elif FLAGS.criteria=='precision':
            test_cri=overall_precision
        elif FLAGS.criteria=='recall':
            test_cri=overall_recall
        elif FLAGS.criteria=='f1':
            test_cri=overall_f1
        elif FLAGS.criteria=='bacc':
            test_cri=overall_bacc
        elif FLAGS.criteria=='roc_auc':
            test_cri=overall_roc_auc

        for key in key_list:
            for cri in ['accuracy', 'precision','recall','f1','bacc','roc_auc']:
                if test_grp_cri[key][cri]>best_grp_cri[key][cri]:
                    best_grp_cri[key][cri]=test_grp_cri[key][cri]

        if test_cri>best_overall_cri:
            saved=True
            best_overall_cri=test_cri
            overall_grp_cri=test_grp_cri
            overall['accuracy']=overall_acc
            overall['precision']=overall_precision
            overall['recall']=overall_recall
            overall['f1']=overall_f1
            overall['bacc']=overall_bacc
            overall['roc_auc']=overall_roc_auc

            torch.save(shared_expert.state_dict(),os.path.join(model_dir,"shared_expert-%.4d.pt"%para_id))
            torch.save(share_lin_attn.state_dict(),os.path.join(model_dir,"share_lin_attn-%.4d.pt"%para_id))
            torch.save(share_attention.state_dict(),os.path.join(model_dir,"share_attention-%.4d.pt"%para_id))
            if FLAGS.shared_output_flag:
                torch.save(output_layers.state_dict(),os.path.join(model_dir,"output_layers-%.4d.pt"%para_id))
            for key in key_list:
                for n,c in dataset_dict[key]['model']._modules.items():
                    if 'share' not in n:
                        torch.save(c.state_dict(),os.path.join(model_dir,key+"_"+n+"-%.4d.pt"%para_id))

    # %%
    results_dict['best_overall_'+FLAGS.criteria].append(best_overall_cri)

    # %%
    for cri in ['accuracy', 'precision','recall','f1','bacc','roc_auc']:
        results_dict['overall_'+cri].append(overall[cri])
        tmp=[]
        for key in key_list:
            results_dict['best_'+key+'_'+cri].append(best_grp_cri[key][cri])
            results_dict['best_overall_'+key+'_'+cri].append(overall_grp_cri[key][cri])
            tmp.append(overall_grp_cri[key][cri])
        results_dict['delta_'+cri].append(max(tmp)-min(tmp))
        if overall[cri]!=0 and overall[cri] is not np.nan:
            results_dict['relative_delta_'+cri].append(results_dict['delta_'+cri][-1]/overall[cri])
        else:
            results_dict['relative_delta_'+cri].append(np.nan)

    # %%
    if not saved:
        results_dict["supp_overall_"+FLAGS.criteria].append(results_dict["best_overall_"+FLAGS.criteria][-1])
        for cri in ['accuracy', 'precision','recall','f1','bacc','roc_auc']:
            results_dict['supp_'+cri].append(overall[cri])
            for key in key_list:
                results_dict['supp_'+key+'_'+cri].append(results_dict['best_overall_'+key+'_'+cri][-1])
            results_dict['supp_delta_'+cri].append(results_dict['delta_'+cri][-1])
            results_dict['supp_relative_delta_'+cri].append(results_dict['relative_delta_'+cri][-1])
        return

    # %%
    shared_expert=Expert(input_shape=input_shape, output_shape=output_shape, drop_out=drop_out, shrinking_speed=shrinking_speed, middle_inflation_flag=middle_inflation_flag, middle_inflation=middle_inflation)
    shared_expert.load_state_dict(torch.load(os.path.join(model_dir,"shared_expert-%.4d.pt"%para_id),weights_only=True))
    share_lin_attn=Expert(input_shape, shared_expert.output_features, shrinking_speed=shrinking_speed, drop_out=drop_out, middle_inflation_flag=middle_inflation_flag, middle_inflation=middle_inflation)
    share_lin_attn.load_state_dict(torch.load(os.path.join(model_dir,"share_lin_attn-%.4d.pt"%para_id),weights_only=True))
    share_attention=nn.MultiheadAttention(1,1, dropout=drop_out, batch_first=True)
    share_attention.load_state_dict(torch.load(os.path.join(model_dir,"share_attention-%.4d.pt"%para_id),weights_only=True))
    if FLAGS.shared_output_flag:
        output_layers=Expert(shared_expert.output_features+len(split_columns)*output_shape, Y_train.shape[-1], shrinking_speed=shrinking_speed, drop_out=drop_out, middle_inflation_flag=middle_inflation_flag, middle_inflation=middle_inflation, classification_flag=True)
        output_layers.load_state_dict(torch.load(os.path.join(model_dir,"output_layers-%.4d.pt"%para_id),weights_only=True))
        shared_expert_optim=torch.optim.Adam([
            {'params': shared_expert.parameters()},
            {'params': share_lin_attn.parameters()},
            {'params': share_attention.parameters()},
            {'params': output_layers.parameters()}
        ],lr=base_lr/2,weight_decay=16)
    else:
        shared_expert_optim=torch.optim.Adam([
            {'params': shared_expert.parameters()},
            {'params': share_lin_attn.parameters()},
            {'params': share_attention.parameters()}
        ],lr=base_lr/2,weight_decay=16)

    # %%
    for key in key_list:
        dataset_dict[key]['conf_expert']=Expert(input_shape=input_shape, output_shape=output_shape, drop_out=drop_out, shrinking_speed=shrinking_speed, middle_inflation_flag=middle_inflation_flag, middle_inflation=middle_inflation)
        if FLAGS.shared_output_flag:
            dataset_dict[key]['model']=TypeAttnNet(input_shape=input_shape, shared_expert= shared_expert, share_lin_attn= share_lin_attn, share_attention= share_attention, conf_experts= nn.ModuleList([dataset_dict[key]['conf_expert']]), shared_output_flag=FLAGS.shared_output_flag, num_classes=Y_train.shape[-1], output_layers=output_layers, num_heads=num_heads, drop_out=drop_out, shrinking_speed=shrinking_speed)
        else:
            dataset_dict[key]['model']=TypeAttnNet(input_shape=input_shape, shared_expert= shared_expert, share_lin_attn= share_lin_attn, share_attention= share_attention, conf_experts= nn.ModuleList([dataset_dict[key]['conf_expert']]), shared_output_flag=FLAGS.shared_output_flag, num_classes=Y_train.shape[-1], num_heads=num_heads, drop_out=drop_out, shrinking_speed=shrinking_speed)
        for n,c in dataset_dict[key]['model']._modules.items():
            if 'share' not in n:
                c.load_state_dict(torch.load(os.path.join(model_dir,key+"_"+n+"-%.4d.pt"%para_id),weights_only=True))
        shared_expert_params = [p for name, p in dataset_dict[key]['model'].named_parameters() if 'share' in name]
        others = [p for name, p in dataset_dict[key]['model'].named_parameters() if 'share' not in name]
        dataset_dict[key]['optimizer']=torch.optim.Adam([
            {'params':others},
            {'params':shared_expert_params, 'lr':0}
        ],lr=base_lr*(minimum_class/len(dataset_dict[key]['df']))/2,weight_decay=16)

    # %%
    tmp_cri=[_[FLAGS.criteria] for _ in overall_grp_cri.values()]
    if max(tmp_cri)-min(tmp_cri)>0.02:
        to_train=np.argmin(tmp_cri)

        max_iterations=5000

        t = 0
        loss=1
        smallest_diff=max(tmp_cri)-min(tmp_cri)
        supp={'accuracy':0, 'precision':0, 'recall':0, 'f1':0, 'bacc':0,'roc_auc':0}
        while loss>0:
            shared_expert.eval()
            share_lin_attn.eval()
            share_attention.eval()
            shared_expert.to(device)
            for i in range(len(key_list)):
                if i!=to_train:
                    continue
                key=key_list[i]
                dataloader_train_surv = dataset_dict[key]['train_dataloader']
                surv_model = dataset_dict[key]['model']

                surv_model.train()
                surv_model.to(device)
                for batch, (X, Y) in enumerate(dataloader_train_surv):
                    X=X.to(device)
                    Y=Y.to(device)

                    pred = surv_model(X)
                    loss = loss_fn(pred, Y)
                
            loss.backward()
            for i in range(len(key_list)):
                if i!=to_train:
                    continue
                key=key_list[i]
                opt = dataset_dict[key]['optimizer']
                opt.step()
            shared_expert_optim.zero_grad()
            for i in range(len(key_list)):
                key=key_list[i]
                opt = dataset_dict[key]['optimizer']
                opt.zero_grad()

            for key in key_list:
                dataloader_train_surv = dataset_dict[key]['train_dataloader']
                model = dataset_dict[key]['model']
                key_loss, key_cm, (key_acc, key_precision, key_recall, key_f1, key_bacc, key_roc_auc), (key_y_np, key_pred_np) = test_type(dataloader_train_surv, model, loss_fn, "train", device)

            test_grp_cri={}
            cms=np.zeros(key_cm.shape)
            for key in key_list:
                test_grp_cri[key]={}
                dataloader_test_surv = dataset_dict[key]['test_dataloader']
                model = dataset_dict[key]['model']
                tmp_loss, tmp_cm, (tmp_acc, tmp_precision, tmp_recall, tmp_f1, tmp_bacc, tmp_roc_auc), (tmp_y_np, tmp_pred_np) = test_type(dataloader_test_surv, model, loss_fn, "test", device)
                test_grp_cri[key]['accuracy']=tmp_acc
                test_grp_cri[key]['precision']=tmp_precision
                test_grp_cri[key]['recall']=tmp_recall
                test_grp_cri[key]['f1']=tmp_f1
                test_grp_cri[key]['bacc']=tmp_bacc
                test_grp_cri[key]['roc_auc']=tmp_roc_auc
                test_grp_cri[key]['y_np']=tmp_y_np
                test_grp_cri[key]['pred_np']=tmp_pred_np
                cms+=tmp_cm
            
            try:
                overall_acc, overall_precision, overall_recall, overall_f1, overall_bacc=cm_to_metrics(cms)
            except IndexError:
                if cms[1,1]==0:
                    overall_acc, overall_precision, overall_recall, overall_f1, overall_bacc=1,np.nan,np.nan,np.nan,np.nan
                else:
                    overall_acc, overall_precision, overall_recall, overall_f1, overall_bacc=1,1,1,1,np.nan
            try:
                overall_y_np=np.concatenate([test_grp_cri[key]['y_np'] for key in key_list])
                overall_pred_np=np.concatenate([test_grp_cri[key]['pred_np'] for key in key_list])
                overall_roc_auc=roc_auc_score(overall_y_np,overall_pred_np)
            except ValueError:
                overall_roc_auc=np.nan
            if FLAGS.criteria=='accuracy':
                test_cri=overall_acc
            elif FLAGS.criteria=='precision':
                test_cri=overall_precision
            elif FLAGS.criteria=='recall':
                test_cri=overall_recall
            elif FLAGS.criteria=='f1':
                test_cri=overall_f1
            elif FLAGS.criteria=='bacc':
                test_cri=overall_bacc
            elif FLAGS.criteria=='roc_auc':
                test_cri=overall_roc_auc

            tmp_cri=[_[FLAGS.criteria] for _ in test_grp_cri.values()]
            tmp_diff=max(tmp_cri)-min(tmp_cri)

            if all([test_grp_cri[_][FLAGS.criteria]>=overall_grp_cri[_][FLAGS.criteria] for _ in key_list]) and test_cri>=best_overall_cri and tmp_diff<=smallest_diff:
                for n,c in dataset_dict[key_list[to_train]]['model']._modules.items():
                    if 'share' not in n:
                        torch.save(c.state_dict(),os.path.join(model_dir,"%s_supp"%(key_list[to_train])+"_"+n+"-%.4d.pt"%para_id))

                best_overall_cri=test_cri
                overall_grp_cri=test_grp_cri
                smallest_diff=tmp_diff
                supp['accuracy']=overall_acc
                supp['precision']=overall_precision
                supp['recall']=overall_recall
                supp['f1']=overall_f1
                supp['bacc']=overall_bacc
                supp['roc_auc']=overall_roc_auc
                if smallest_diff<0.02:
                    break

            if t>=max_iterations:
                break

            to_train=np.argmin(tmp_cri)
            t+=1
        
        results_dict["supp_overall_"+FLAGS.criteria].append(best_overall_cri)
        for cri in ['accuracy', 'precision','recall','f1','bacc','roc_auc']:
            results_dict['supp_'+cri].append(supp[cri])
            tmp=[]
            for key in key_list:
                results_dict['supp_'+key+'_'+cri].append(overall_grp_cri[key][cri])
                tmp.append(overall_grp_cri[key][cri])
            results_dict['supp_delta_'+cri].append(max(tmp)-min(tmp))
            if supp[cri]!=0 and supp[cri] is not np.nan:
                results_dict['supp_relative_delta_'+cri].append(results_dict['supp_delta_'+cri][-1]/supp[cri])
            else:
                results_dict['supp_relative_delta_'+cri].append(np.nan)
    else:
        results_dict["supp_overall_"+FLAGS.criteria].append(results_dict["best_overall_"+FLAGS.criteria][-1])
        for cri in ['accuracy', 'precision','recall','f1','bacc','roc_auc']:
            results_dict['supp_'+cri].append(overall[cri])
            for key in key_list:
                results_dict['supp_'+key+'_'+cri].append(results_dict['best_overall_'+key+'_'+cri][-1])
            results_dict['supp_delta_'+cri].append(results_dict['delta_'+cri][-1])
            results_dict['supp_relative_delta_'+cri].append(results_dict['relative_delta_'+cri][-1])
    
    result_queue.put(results_dict)

# %%
def stratify_age(x: np.ndarray, middle='median'):
    if middle=='median':
        middle=np.median(x)
    elif middle=='mean':
        middle=np.average(x)
    results=[]
    for i in x:
        if i<=middle:
            results.append("le%.2f"%middle) # less than or equal to
        else:
            results.append("gt%.2f"%middle) # greater than
    return results, middle

# %%
def main(FLAGS):
    global device

    mp.set_start_method('spawn')

    # Get cpu, gpu or mps device for training.
    if torch.cuda.is_available():
        device = torch.device(
            "cuda:"+FLAGS.gpu_device
        )
    else:
        device = "cpu"
    print(f"Using {device} device")

    random_states=[5,10]
    middle_inflation_rates=[5,6]
    output_shapes=[10,15]
    shrinking_speeds=[2,3]
    drop_outs=[0.1,0.2]
    middle_inflation_flag=True
    num_heads=1
    base_lr=1e-4
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

    assert FLAGS.gene_list in gene_list_dict.keys(), "invalid value for gene_list"
    gene_list_name=FLAGS.gene_list
    gene_list=gene_list_dict[gene_list_name]

    switch_data_source=FLAGS.data_source

    assert FLAGS.criteria in ['accuracy', 'precision','recall','f1','bacc','roc_auc'], "invalid criteria"

    # %%
    df=pd.read_csv(os.path.join("..","data","ASD","joined-%s.tsv"%(switch_data_source)),sep='\t',header=0,index_col=0, low_memory=False)

    # %%
    print(np.sum(df.isna().to_numpy()))

    # %%
    columns=list(df.columns)

    # %%
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
    print(columns[-5:])

    # %%
    split_columns=[
        FLAGS.factor
    ]
    assert FLAGS.factor in df.columns, "specified factor "+FLAGS.factor+" not presented in data source!"

    df=df.loc[df[FLAGS.factor].notna()&df[FLAGS.target_col].notna(),:]

    # %%
    i=0
    dir_name="%s_%s_%s_%.4d"%(FLAGS.factor,switch_data_source,FLAGS.criteria,i)
    dir_name+=("_"+gene_list_name)
    while os.path.exists(dir_name):
        i+=1
        dir_name="%s_%s_%s_%.4d"%(FLAGS.factor,switch_data_source,FLAGS.criteria,i)
        dir_name+=("_"+gene_list_name)
    os.mkdir(dir_name)

    # %%
    i=0
    if not os.path.exists("models"):
        os.mkdir("models")
    model_dir=os.path.join("models","%s_%s_%s_%.4d"%(FLAGS.factor,switch_data_source,FLAGS.criteria,i))
    model_dir+=("_"+gene_list_name)
    while os.path.exists(model_dir):
        i+=1
        model_dir=os.path.join("models","%s_%s_%s_%.4d"%(FLAGS.factor,switch_data_source,FLAGS.criteria,i))
        model_dir+=("_"+gene_list_name)
    os.mkdir(model_dir)

    # %%
    df=df[gene_list+split_columns+[FLAGS.target_col]]

    # %%
    if 'age' in FLAGS.factor:
        df.loc[:,FLAGS.factor], _ =stratify_age(df[FLAGS.factor].to_numpy(),middle='median')
    
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
    if np.sum(df.isna().to_numpy())>0:
        print("found NaN!")
        print("original shape: "+str(df.shape))
        df=df.dropna(axis=0)
    print("shape: "+str(df.shape))

    if FLAGS.target_col=='diagnosis':
        df.loc[df['diagnosis']=='normal','diagnosis']=0
        df.loc[df['diagnosis']=='diabetes','diagnosis']=1
        df['diagnosis']=df['diagnosis'].astype(int)

    # %%
    single_parameters=[]
    for random_state in random_states:
        for shrinking_speed in shrinking_speeds:
            for drop_out in drop_outs:
                for middle_inflation in middle_inflation_rates:
                    single_parameters.append((random_state, shrinking_speed, drop_out, middle_inflation))
    
    # %%
    with mp.Manager() as manager:
        # Create a queue to communicate results
        result_queue = manager.Queue()

        # Start the collector process
        collector_process = mp.Process(target=collector_single, args=(dir_name, result_queue,len(single_parameters),))
        collector_process.start()

        # Create a pool of worker processes
        pool = mp.Pool(processes=FLAGS.n_parallel)
        returns=[]

        # Start the worker tasks
        for model_id in range(len(single_parameters)):
            returns.append(pool.apply_async(worker_single, args=(model_id, FLAGS, device, middle_inflation_flag, base_lr, single_parameters, gene_list, split_columns, columns, df.copy(), result_queue,)))

        # Close the pool and wait for all tasks to complete
        pool.close()
        pool.join()

        # Send a stop signal to the collector process
        result_queue.put("STOP")

        # Wait for the collector process to finish
        print("Main process waiting for collector to finish")
        collector_process.join()

        for _ in returns:
            msg=_.get()
            if msg!=None:
                print(msg)
                exit(1)

        print("single mode done")

    # %% [markdown]
    # # multi-expert

    # %%
    dataset_dict={}
    for grp in df[FLAGS.factor].unique():
        dataset_dict[grp]={}
        indices=df[df[FLAGS.factor]==grp].index
        dataset_dict[grp]['df']=df.loc[indices].copy()
    key_list=list(dataset_dict.keys())

    # %%
    sizes=[len(x['df']) for x in dataset_dict.values()]
    minimum_class=min(sizes)

    # %%
    print(key_list)
    print(sizes)

    # %%
    multi_parameters=[]
    for random_state in random_states:
        for output_shape in output_shapes:
            for drop_out in drop_outs:
                for shrinking_speed in shrinking_speeds:
                    for middle_inflation in middle_inflation_rates:
                        multi_parameters.append((random_state,output_shape,drop_out,shrinking_speed,middle_inflation))

    # %%
    with mp.Manager() as manager:
        # Create a queue to communicate results
        result_queue = manager.Queue()

        # Start the collector process
        collector_process = mp.Process(target=collector_multi, args=(dir_name, result_queue,len(multi_parameters),))
        collector_process.start()

        # Create a pool of worker processes
        pool = mp.Pool(processes=FLAGS.n_parallel)
        returns=[]

        # Start the worker tasks
        for para_id in range(len(multi_parameters)):
            returns.append(pool.apply_async(worker_multi, args=(para_id, FLAGS, device, multi_parameters, dataset_dict.copy(), model_dir, minimum_class, gene_list, columns, split_columns, key_list, middle_inflation_flag, num_heads, base_lr, result_queue,)))

        # Close the pool and wait for all tasks to complete
        pool.close()
        pool.join()

        # Send a stop signal to the collector process
        result_queue.put("STOP")

        # Wait for the collector process to finish
        print("Main process waiting for collector to finish")
        collector_process.join()

        for _ in returns:
            msg=_.get()
            if msg!=None:
                print(msg)
                exit(1)

        print("multi mode done")

    # %% [markdown]
    # ## supplemental training

# %%
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--gene_list",
        type=str,
        required=True
    )
    parser.add_argument(
        '--data_source',
        type=str,
        required=True
    )
    parser.add_argument(
        '--shared_output_flag',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--middle_inflation',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--gpu_device',
        type=str,
        default="0"
    )
    parser.add_argument(
        '--criteria',
        type=str,
        default="accuracy"
    )
    parser.add_argument(
        '--factor',
        type=str
    )
    parser.add_argument(
        '--target_col',
        type=str
    )
    parser.add_argument(
        '--n_parallel',
        type=int,
        default=6
    )
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
