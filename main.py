# -*- coding: utf-8 -*-
import os
import os.path
import math
import gc
import argparse
import numpy as np
import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable, grad
from model import KT_backbone
from dataset import DATA, PID_DATA
from sklearn.metrics import roc_auc_score
from utils import KTLoss, _l2_normalize_adv
from pytorchtools import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train KT')
    parser.add_argument('--max_iter', type=int, default=150, help='number of iterations')
    parser.add_argument('--seed', type=int, default=224, help='default seed')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
    parser.add_argument('--lr-decay', type=int, default=50, help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--hidden-emb-dim', type=int, default=80, help='Dimension of concept embedding.')
    parser.add_argument('--skill-emb-dim', type=int, default=256)
    parser.add_argument('--answer-emb-dim', type=int, default=96)
    parser.add_argument('--dataset', type=str, default="assist2015")
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--epsilon', type=float, default=10)
    params = parser.parse_args()
    dataset = params.dataset
    
    if dataset in {"statics"}:
        params.n_skill = 1223
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'dataset/'+dataset
        params.data_name = dataset
        params.beta = 0.2
        params.epsilon=12

    if dataset in {"assist2009_pid"}:
        params.n_skill = 110
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'dataset/'+dataset
        params.data_name = dataset
        params.beta = 0.2
        params.epsilon=10
        
    if dataset in {"assist2015"}:
        params.n_skill = 100
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'dataset/'+dataset
        params.data_name = dataset
        params.beta = 1
        params.epsilon=15
        
    if dataset in {"assist2017_pid"}:
        params.n_skill = 102
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'dataset/'+dataset
        params.data_name = dataset
        params.beta = 1
        params.epsilon=12

    # Seed Setup
    seedNum = params.seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)


    log_file = os.path.join('{}_test_result.txt'.format(params.data_name))
    log = open(log_file, 'w')
    
    auc_test_list=[]
    
    now=time.time()

    for dataset_set_index in range(5):
        params.dataset_set_index=dataset_set_index+1
    
        # model
        net = KT_backbone(params.skill_emb_dim, params.answer_emb_dim, params.hidden_emb_dim, params.n_skill)
        net=net.to(device)
        
        # optimizer
        optimizer = optim.Adam(net.parameters(), lr=params.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=params.lr_decay, gamma=params.gamma)
        
        # loss Function 
        kt_loss = KTLoss()
        
        # dataset
        if "pid" not in params.data_name:
            dat = DATA(n_question=params.n_skill,
                       seqlen=params.seqlen, separate_char=',', maxstep=500)
        else:
            dat = PID_DATA(n_question=params.n_skill,
                           seqlen=params.seqlen, separate_char=',', maxstep=500)
    
        train_data_path = params.data_dir + "/" + \
            params.data_name + "_train"+str(params.dataset_set_index)+".csv"
        valid_data_path = params.data_dir + "/" + \
            params.data_name + "_valid"+str(params.dataset_set_index)+".csv"
        test_data_path = params.data_dir + "/" + \
            params.data_name + "_test"+str(params.dataset_set_index)+".csv"

        train_skill_data, train_answer_data = dat.load_data(train_data_path)
        val_skill_data, val_answer_data = dat.load_data(valid_data_path)
        test_skill_data,  test_answer_data = dat.load_data(test_data_path)

        # early stopping
        early_stopping = EarlyStopping(patience=20, verbose=True)
        
        # train and validation
        for epoch in range(params.max_iter):
            
            shuffled_ind = np.arange(train_skill_data.shape[0])
            np.random.shuffle(shuffled_ind)
            train_skill_data = train_skill_data[shuffled_ind, :]
            train_answer_data = train_answer_data[shuffled_ind, :]
            
            net.train()
            
            y_true_train_list = []
            y_pred_train_list = []
            train_N = int(math.ceil(len(train_skill_data) / params.batch_size))
    
            for idx in range(train_N):
                optimizer.zero_grad()
                train_batch_skill   = train_skill_data[idx*params.batch_size:(idx+1)*params.batch_size]
                train_batch_answer  = train_answer_data[idx*params.batch_size:(idx+1)*params.batch_size]
                
                skill = torch.LongTensor(train_batch_skill)
                answer = torch.LongTensor(train_batch_answer)
                skill = torch.where(skill==-1, torch.tensor([params.n_skill]), skill)
                answer = torch.where(answer==-1, torch.tensor([2]), answer)
                skill, answer = skill.to(device),answer.to(device)
      
                pred_res, features = net(skill, answer)
                loss, y_pred, y_true = kt_loss(pred_res, answer)
         
                features_grad = grad(loss, features, retain_graph=True)
                p_adv = torch.FloatTensor(params.epsilon * _l2_normalize_adv(features_grad[0].data))
                p_adv = Variable(p_adv).to(device)
                pred_res, features = net(skill, answer, p_adv)
                adv_loss, _ , _ = kt_loss(pred_res, answer)

                total_loss=loss+ params.beta*adv_loss
                total_loss.backward()
                optimizer.step()
                
                y_pred_train_list.append(y_pred.cpu().detach().numpy())
                y_true_train_list.append(y_true.cpu().detach().numpy())
                
            scheduler.step()
                
            all_y_pred_train = np.concatenate(y_pred_train_list, axis=0)
            all_y_true_train = np.concatenate(y_true_train_list, axis=0)

            auc_train = roc_auc_score(all_y_true_train, all_y_pred_train)
            print('train epoch: ', (epoch+1), 'train auc: ', auc_train)

            val_total_loss = []
            y_true_val_list = []
            y_pred_val_list = []
            val_N = int(math.ceil(len(val_skill_data) / params.batch_size))
            net.eval()
            with torch.no_grad():
                for idx in range(val_N):
                    val_batch_skill   = val_skill_data[idx*params.batch_size:(idx+1)*params.batch_size]
                    val_batch_answer  = val_answer_data[idx*params.batch_size:(idx+1)*params.batch_size]
                    
                    skill=torch.LongTensor(val_batch_skill)
                    answer=torch.LongTensor(val_batch_answer)
                    skill = torch.where(skill==-1, torch.tensor([params.n_skill]), skill)
                    answer = torch.where(answer==-1, torch.tensor([2]), answer)
                    skill,answer=skill.to(device),answer.to(device)
                    
                    pred_res, features = net(skill, answer)
                    loss, y_pred, y_true = kt_loss(pred_res, answer)
                    
                    val_total_loss.append(loss.item())
                    y_pred_val_list.append(y_pred.cpu().detach().numpy())
                    y_true_val_list.append(y_true.cpu().detach().numpy())
                    
                all_y_pred_val = np.concatenate(y_pred_val_list, axis=0)
                all_y_true_val = np.concatenate(y_true_val_list, axis=0)
           
                auc_val = roc_auc_score(all_y_true_val, all_y_pred_val)
                print('val epoch: ', (epoch+1), 'val loss: ', loss.item(), 'val auc: ', auc_val)

                save_model_file = os.path.join('./fold{}'.format(params.dataset_set_index))
                if not os.path.exists(save_model_file):
                    os.mkdir(save_model_file)
                    
                early_stopping(np.average(val_total_loss), net, save_path=os.path.join(save_model_file, 'kt_model_best.pt'))
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                del auc_train
                del auc_val
                gc.collect()
                torch.cuda.empty_cache()
                
        # test
        load_model_path=os.path.join(save_model_file, 'kt_model_best.pt')
        net.load_state_dict(torch.load(load_model_path))
        net.eval()
        y_true_test_list = []
        y_pred_test_list = []
        test_N = int(math.ceil(len(test_skill_data) / params.batch_size))
        with torch.no_grad():
            for idx in range(test_N):
                test_batch_skill   = test_skill_data[idx*params.batch_size:(idx+1)*params.batch_size]
                test_batch_answer  = test_answer_data[idx*params.batch_size:(idx+1)*params.batch_size]
                
                skill=torch.LongTensor(test_batch_skill)
                answer=torch.LongTensor(test_batch_answer)
                skill = torch.where(skill==-1, torch.tensor([params.n_skill]), skill)
                answer = torch.where(answer==-1, torch.tensor([2]), answer)
                skill,answer=skill.to(device),answer.to(device)
                
                pred_res, features = net(skill, answer)
                loss, y_pred, y_true = kt_loss(pred_res, answer)
                
                y_true_test_list.append(y_true.cpu().detach().numpy())
                y_pred_test_list.append(y_pred.cpu().detach().numpy())
                
            all_y_true_test = np.concatenate(y_true_test_list, 0)
            all_y_pred_test = np.concatenate(y_pred_test_list, 0)

            auc_test = roc_auc_score(all_y_true_test, all_y_pred_test)
            
            print('fold{}'.format(params.dataset_set_index), 'test auc: ', auc_test)
            auc_test_list.append(auc_test)
            print('fold{}'.format(params.dataset_set_index), 'test auc: ', auc_test, file=log)

            del auc_test
            gc.collect()
            torch.cuda.empty_cache()
    print('average test auc:', np.round(np.mean(auc_test_list), decimals = 4), u'\u00B1', np.round(np.std(auc_test_list), decimals = 4))
    print('average test auc:', np.round(np.mean(auc_test_list), decimals = 4), u'\u00B1', np.round(np.std(auc_test_list), decimals = 4), file=log)
    
    del auc_test_list
    log.close()
    
    end=time.time()
    print('total running time:{} min'.format((end-now)/60))
