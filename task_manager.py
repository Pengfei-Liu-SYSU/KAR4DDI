# -*- coding: utf-8 -*-
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import logging
logger = logging.getLogger(__name__)

import warnings

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn import metrics

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from argparse import ArgumentParser
from models.model_manager import MtiModel
from utils.xutils import print_model_info, custom_collate_fn, ToDevice
from datasets.dataset_manager import MtiDataset
from torch.utils.data import DataLoader

import datetime
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import json
from utils import AverageMeter
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from models import SUPPORTED_Tokenizer, SUPPORTED_CKPT
tqdm.pandas(desc="Processing")

import random

import pathlib
HERE = pathlib.Path(__file__).resolve().parent
print(HERE)




parser = ArgumentParser(description='Training.')
parser.add_argument('--batch_size', default=24, type=int, metavar='N', help='')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument('--device',type=int,default=2,help = 'the number of GPUS')
parser.add_argument('--script_mode', type=str, default ='train', help='data_check/data_label/model_check/encoder_check/train/eval')
parser.add_argument('--model_name', type=str, default ='Biot5', help='Biot5')
parser.add_argument('--input_modal', type=str, default ='type_desc', help='type_selfies/desc/selfies')
parser.add_argument('--freeze', type=bool, default =False, help='model freeze')
parser.add_argument("--model_pretrain", type=str, default="./ckpts/text_ckpts/biot5")
parser.add_argument("--task", type=str, default="cluster_b_k_a_rl", help="")
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--model_output_path", type=str, default="output/")
parser.add_argument("--log_save_path", type=str, default="log")
parser.add_argument("--result_save_path", type=str, default="result")
parser.add_argument("--latest_checkpoint", type=str, default="ckpts/finetune_ckpts")
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--logging_steps", type=int, default=300)
parser.add_argument("--ckpt_output_path", type=str, default="ckpts/finetune_ckpts")
parser.add_argument("--patience", type=int, default=2)
parser.add_argument("--toy", type=bool, default=False)
parser.add_argument("--dataset", type=str, default='deepddie/few_k_b_a_5_20/')
parser.add_argument("--dataset_folder", type=str, default='data/deepddie/few_k_b_a_5_20/')
parser.add_argument("--sub", type=str, default='kmeans_9')
parser.add_argument("--shot", type=str, default='all', help="com/few/rare")



args = parser.parse_args()

print("start----------------------------------\n")



def roc_auc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def pr_auc(y, pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc


def train_mti_decoder(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss = None):
    running_loss = AverageMeter()
    step = 0
    loss_values = {"train_loss": [], "valid_loss": [], "test_loss": []}
    last_ckpt_file = None
    patience = 0
    device = args.device
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        #model.train()
        train_loss = []
        train_loader = tqdm(train_loader, desc="Training")
        for mol in train_loader:
            mol = ToDevice(mol, args.device)
            loss = model(mol)
            #accelerator.backward(loss)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                train_loss.append(running_loss.get_average())
                running_loss.reset()
        loss_values["train_loss"].append(np.mean(train_loss))
        val_loss = val_mti_decoder(valid_loader, model, device)
        #test_loss = val_mti_decoder(test_loader, model, device)
        test_loss = 0
        loss_values["valid_loss"].append(val_loss)
        loss_values["test_loss"].append(test_loss)

        if best_loss == None or best_loss-val_loss>0.001 :
            patience = 0
            best_loss = val_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            print(loss_values)
        else:
            patience = patience+1
            scheduler.step()
            message = f"epoch: {epoch}, best_loss: {best_loss} ,val_loss: {val_loss}, ckpt passed, patience : {patience}. "

            print(loss_values)
            
        if patience > args.patience or best_loss<0.0001:
            message = f"epoch: {epoch}, best_loss: {best_loss} ,val_loss: {val_loss}, ckpt passed, patience : {patience}. "
            acc, pre_score, f1 = test_mti_decoder(test_loader, model, device, message)
            message = message + f"epoch {epoch-1}, metric -- acc: {acc}, pre_score : {pre_score}, f1 :{f1}."
            print("Early stopping due to reaching patience limit.")
            break
            
    return acc, f1

def _eval(y_pred, labels):

    y_pred_numeric = y_pred
    labels_numeric = labels

    acc = accuracy_score(labels_numeric, y_pred_numeric)
    pre_score = precision_score(labels_numeric, y_pred_numeric)
    f1 = f1_score(labels_numeric, y_pred_numeric)

    return acc, pre_score, f1  

def __eval(truth_list, result_list):
    
    truth_list_int = [int(item) for item in truth_list]


    roc_auc = roc_auc_score(truth_list_int, result_list)
    print(f"ROC AUC Score: {roc_auc}")
    threshold = 0.5
    predicted_labels = [1 if prob[0] > threshold else 0 for prob in result_list]


    accuracy = sum([pred == true for pred, true in zip(predicted_labels, truth_list_int)]) / len(truth_list_int)
    print(f"Accuracy: {accuracy}")
    precision = precision_score(truth_list_int, predicted_labels)
    recall = recall_score(truth_list_int, predicted_labels)
    f1 = f1_score(truth_list_int, predicted_labels)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    return accuracy, precision, f1 , roc_auc, recall

def preprocess_result_list(input_list):
    
    processed_list = []
    for item in input_list:
        try:
            processed_list.append(int(item))
        except ValueError:
            
            processed_list.append(113)
    return processed_list

def __eval_multi_label(truth_list, result_list, n_classes=113):
 
    result_list = preprocess_result_list(result_list)
    truth_list_binarized = label_binarize(truth_list, classes=list(range(n_classes)))
    result_list_binarized = label_binarize(result_list, classes=list(range(n_classes)))
    

    print(f"truth_list_binarized: {truth_list_binarized}")
    print(f"result_list_binarized: {result_list_binarized}")

    
    accuracy = accuracy_score(truth_list, result_list)
    print(f"Accuracy: {accuracy}")
    
    precision = precision_score(truth_list, result_list, average='macro')
    recall = recall_score(truth_list, result_list, average='macro')
    f1 = f1_score(truth_list, result_list, average='macro')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    return accuracy, precision, f1 , recall

def val_mti_decoder(valid_loader, model, device):
    model.eval()
    val_loss = 0
    logger.info("Validating...")
    with torch.no_grad():
        valid_loader = tqdm(valid_loader, desc="Validation")
        for i, mol in enumerate(valid_loader):
            mol = ToDevice(mol, device)
            truth = mol['truth']
            loss = model(mol)
            if(i==1):
                result = model.generate_text(mol)
                #result = model.predict(mol)
                print(f"label:{truth[0]} | Result : {result[0]}")
            val_loss += loss.detach().cpu().item()
        logger.info("validation loss %.4lf" % (val_loss / len(valid_loader)))
    return val_loss / len(valid_loader)

def test_mti_decoder(test_loader, model, device, message = None):
    model.eval()
    test_loss = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    logger.info("Testing...")
    i = 0
    with torch.no_grad():
        test_loader = tqdm(test_loader, desc="Test")
        truth_list = []
        result_list = []
        id_list = []
        for mol in test_loader:
            mol = ToDevice(mol, device)

            truth = mol['truth']
            truth_list = truth_list + truth

            result = model.generate_text(mol)
            if(i==1):
                print(f"label:{truth[0]} | Result : {result[0]}")
       
            i=i+1
            result_list = result_list + result

        assert len(truth_list) == len(result_list)
        acc, pre_score, f1, recall = __eval_multi_label(truth_list, result_list)
        message_1 = f"dataset metric 1: {args.dataset}, acc: {acc}, pre_score : {pre_score}, f1 : {f1}, recall: {recall}, loss: {message}"
        print(message_1)
        return acc, pre_score, f1




clustering_methods = ['kmeans', 'birch', 'agglomerative']
modalities = ['type_selfies', 'type_desc']    
learning_rates = [0.0005, 0.00075, 0.001]
batch_sizes = [12, 18, 24]

def get_valid_action(current_action, num_actions):
    valid_actions = list(range(num_actions))
    valid_actions.remove(current_action)
    return random.choice(valid_actions)

def random_initial_state():
    cluster_name = random.choice(clustering_methods) 
    n_clusters = random.randint(5, 20)  
    modal_name = random.choice(modalities)  
    lr = random.choice(learning_rates)  
    bs = random.choice(batch_sizes)  

    return encode_state(cluster_name, n_clusters, modal_name, lr, bs)

def evaluate_with_rl(args, episodes=10):
    best_accuracy = 0
    best_f1 = 0
    best_params = None
    results = []
    num_clustering_methods = len(clustering_methods)
    num_clusters = 16
    num_modality = len(modalities)
    num_actions = 8

    q_table = np.zeros((num_clustering_methods * num_clusters * num_modality * len(learning_rates) * len(batch_sizes), num_actions))
    alpha = 0.1  # learning rate
    gamma = 0.6  # discount factor
    epsilon = 0.3  # exploration rate

    initial_state = random_initial_state()
    state = initial_state

    no_improvement_counter = 0
    no_improvement_threshold = 10
    if not os.path.exists(f"{args.result_save_path}/{args.task}"):
        os.makedirs(f"{args.result_save_path}/{args.task}")
    result_file = f"{args.result_save_path}/{args.task}/rl_{args.shot}.txt"
    
    for episode in range(episodes):
        if(best_params!= None):
            state = encode_state(*best_params)
        done = False
        print(f"---------episode---------: {episode}")
        while not done:
            cluster_name, n_clusters, modal_name, lr, bs = decode_state(state)
            action = get_action(state, q_table, epsilon)
            print(f"Initial Action: {action}")
            attempts = 0
            while attempts < 10:  # To prevent infinite loops
                if action == 0:
                    if n_clusters > 5:
                        n_clusters -= 1
                        print(f"New cluster_num: {n_clusters}")
                        break
                    else:
                        action = get_valid_action(action, 8)
                        print(f"Action replaced: {action}")

                elif action == 1:
                    if n_clusters < 20:
                        n_clusters += 1
                        print(f"New cluster_num: {n_clusters}")
                        break
                    else:
                        action = get_valid_action(action, 8)
                        print(f"Action replaced: {action}")

                elif action == 2:
                    available_cluster_names = [name for name in clustering_methods if name != cluster_name]
                    if available_cluster_names:
                        cluster_name = random.choice(available_cluster_names)
                        print(f"Chose new cluster_name: {cluster_name}")
                        break
                    else:
                        print("No alternative cluster methods available")
                        action = get_valid_action(action, 8)
                        print(f"Action replaced: {action}")

                elif action == 3:
                    current_modal_index = modalities.index(modal_name)
                    if current_modal_index < len(modalities) - 1:
                        next_modal_index = current_modal_index + 1
                        modal_name = modalities[next_modal_index]
                        print(f"Updated modality to: {modal_name}")
                        break
                    else:
                        modal_name = modalities[0]  # Wrapping around
                        print(f"Updated modality to: {modal_name} (wrapped around)")
                        break

                elif action == 4:
                    current_lr_index = learning_rates.index(lr)
                    if current_lr_index < len(learning_rates) - 1:
                        next_lr_index = current_lr_index + 1
                        lr = learning_rates[next_lr_index]
                        print(f"Updated learning rate to: {lr}")
                        break
                    else:
                        action = get_valid_action(action, 8)
                        print(f"Action replaced: {action}")

                elif action == 5:
                    current_lr_index = learning_rates.index(lr)
                    if current_lr_index > 0:
                        next_lr_index = current_lr_index - 1
                        lr = learning_rates[next_lr_index]
                        print(f"Updated learning speed to: {lr}")
                        break
                    else:
                        action = get_valid_action(action, 8)
                        print(f"Action replaced: {action}")

                elif action == 6:
                    current_bs_index = batch_sizes.index(bs)
                    if current_bs_index < len(batch_sizes) - 1:
                        next_bs_index = current_bs_index + 1
                        bs = batch_sizes[next_bs_index]
                        print(f"Updated batch size to: {bs}")
                        break
                    else:
                        action = get_valid_action(action, 8)
                        print(f"Action replaced: {action}")

                elif action == 7:
                    current_bs_index = batch_sizes.index(bs)
                    if current_bs_index > 0:
                        next_bs_index = current_bs_index - 1
                        bs = batch_sizes[next_bs_index]
                        print(f"Updated batch size to: {bs}")
                        break
                    else:
                        action = get_valid_action(action, 8)
                        print(f"Action replaced: {action}")

                attempts += 1

                
            args.sub = f"{cluster_name}_{n_clusters}"
            args.input_modal = modal_name
            args.lr = lr
            args.batch_size = bs
            print(f"{args.sub}, {args.input_modal}, {args.lr}, {args.batch_size}")
            accuracy, f1 = get_rl_result(args)
                                          
            results.append((cluster_name, n_clusters, modal_name, lr, bs, accuracy, f1))
            message_1 = f"Evaluating {cluster_name} with {n_clusters} clusters and {modal_name}, {lr}, {bs}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}"
            print(message_1)

            reward = (accuracy - best_accuracy) + (f1 - best_f1)
            print(f"Reward: {reward}")
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            with open(result_file, 'a') as f:   
                f.write(timestamp+':'+message_1 + ', reward:'+ str(reward) + "\n") 
                
            new_state = encode_state(cluster_name, n_clusters, modal_name, lr, bs)
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state]))

            if accuracy > best_accuracy or (accuracy == best_accuracy and f1 > best_f1):
                best_accuracy = accuracy
                best_f1 = f1
                best_params = (cluster_name, n_clusters, modal_name, lr, bs)
                no_improvement_counter = 0  # Reset the counter on improvement
            else:
                no_improvement_counter += 1
                print(f"no_improvement_counter: {no_improvement_counter}")

            state = new_state

            # Termination condition based on no improvement or state threshold
            if no_improvement_counter >= no_improvement_threshold:
                print(f"No improvement for {no_improvement_threshold} consecutive runs. Returning to best configuration.")
                # Return to best configuration
                state = encode_state(*best_params)
                no_improvement_counter = 0
                done = True  # Break the current loop

        epsilon = max(epsilon * 0.99, 0.1)  # Decrease epsilon over time

    print(f"Best configuration: {best_params} with Accuracy = {best_accuracy:.4f}, F1 = {best_f1:.4f}")
    return results

def encode_state(cluster_name, n_clusters, modal_name, lr, bs):
    cluster_index = clustering_methods.index(cluster_name)
    modal_index = modalities.index(modal_name)
    lr_index = learning_rates.index(lr)
    bs_index = batch_sizes.index(bs)
    
    
    max_clusters_range = 16  # Assuming max range for cluster count (5-20)
    num_lr = len(learning_rates)
    num_bs = len(batch_sizes)
    
    state = (cluster_index * max_clusters_range * len(modalities) * num_lr * num_bs +
             (n_clusters - 5) * len(modalities) * num_lr * num_bs +
             modal_index * num_lr * num_bs +
             lr_index * num_bs +
             bs_index)
    return state

def decode_state(state):
    num_lr = len(learning_rates)
    num_bs = len(batch_sizes)
    max_clusters_range = 16  # Total range for n_clusters (5-20)

    bs_index = state % num_bs
    state //= num_bs
    lr_index = state % num_lr
    state //= num_lr
    modal_index = state % len(modalities)
    state //= len(modalities)
    n_clusters = state % max_clusters_range + 5
    cluster_index = state // max_clusters_range

    cluster_name = clustering_methods[cluster_index]
    modal_name = modalities[modal_index]
    lr = learning_rates[lr_index]
    bs = batch_sizes[bs_index]

    return cluster_name, n_clusters, modal_name, lr, bs

def get_action(state, q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 7)
    else:
        return np.argmax(q_table[state])

def get_rl_result(args):
    
    tokenizer = SUPPORTED_Tokenizer[args.model_name].from_pretrained(SUPPORTED_CKPT[args.model_name])
    tokenizer_org = tokenizer
    model = MtiModel(args)
    print_model_info(model,level=2)
    best_loss = None
    data_ddi_df = pd.read_csv(args.dataset_folder+'k_b_a_5_20_few_226.csv')
    train_mit_dataset = MtiDataset(dataframe = data_ddi_df, split='train', tokenizer_org = tokenizer_org, args = args)
    valid_mit_dataset = MtiDataset(dataframe = data_ddi_df, split='valid', tokenizer_org = tokenizer_org, args = args)
    test_mit_dataset = MtiDataset(dataframe = data_ddi_df, split='test', tokenizer_org = tokenizer_org, args = args)
        
    print(f"train_mit_dataset length: {len(train_mit_dataset)}")
    print(f"valid_mit_dataset length: {len(valid_mit_dataset)}")
    print(f"test_mit_dataset length: {len(test_mit_dataset)}")
        
    train_loader = DataLoader(train_mit_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)

    valid_loader = DataLoader(valid_mit_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_mit_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    model = model.to(args.device)
        
    print(f"now device is {args.device}")
        
    acc, f1 = train_mti_decoder(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss)
    
    return acc, f1

def main():
    args = parser.parse_args()
    device = 'cuda:{}'.format(args.device) if args.device>=0 else 'cpu'
    args.device = device

    tokenizer = SUPPORTED_Tokenizer[args.model_name].from_pretrained(SUPPORTED_CKPT[args.model_name])
    
    if(args.script_mode == 'model_check'):
        model = MtiModel(args)
        print_model_info(model,level=2)
        
    if(args.script_mode == 'eval'):
        args.latest_checkpoint = None
        best_loss = None
        latest_checkpoint = args.latest_checkpoint
        if args.latest_checkpoint:
            print(f"Latest checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoint found.")
        # model
        logger.info("Loading model ......")
        model = MtiModel(args)
        if latest_checkpoint is not None :
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict = False)
        print_model_info(model,level=2)
        logger.info("Loading model successed")
        
        # dataset
        logger.info("Loading dataset ......")

        tokenizer_org = tokenizer
        test_ddi_df = pd.read_csv(args.dataset_folder+'test.csv')
        test_mit_dataset = MtiDataset(dataframe = test_ddi_df, split='test', tokenizer_org = tokenizer_org, args = args)
        test_loader = DataLoader(test_mit_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        logger.info("Loading dataset successed")

        model = model.to(args.device)
        print(f"now device is {args.device}")
        #model, optimizer, test_loader, scheduler = accelerator.prepare(model, optimizer, test_loader, scheduler)
        test_mti_decoder(test_loader, model, args.device)

    if(args.script_mode == 'train'):
        args.latest_checkpoint = None
        best_loss = None
        model = MtiModel(args)
        print_model_info(model,level=2)
        tokenizer_org = tokenizer
        print(f"task: {args.task}")
        data_ddi_df = pd.read_csv(args.dataset_folder+'k_b_a_5_20_few_226.csv')
        train_mit_dataset = MtiDataset(dataframe = data_ddi_df, split='train', tokenizer_org = tokenizer_org, args = args)
        valid_mit_dataset = MtiDataset(dataframe = data_ddi_df, split='valid', tokenizer_org = tokenizer_org, args = args)
        test_mit_dataset = MtiDataset(dataframe = data_ddi_df, split='test', tokenizer_org = tokenizer_org, args = args)
        
        print(f"train_mit_dataset length: {len(train_mit_dataset)}")
        print(f"valid_mit_dataset length: {len(valid_mit_dataset)}")
        print(f"test_mit_dataset length: {len(test_mit_dataset)}")
        
        train_loader = DataLoader(train_mit_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)

        valid_loader = DataLoader(valid_mit_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        test_loader = DataLoader(test_mit_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        model = model.to(args.device)
        
        print(f"now device is {args.device}")
        
        train_mti_decoder(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss)
        
    if(args.script_mode == 'train_rl'):
        args.latest_checkpoint = None
        best_loss = None
        print(f"task: {args.task}")
        evaluate_with_rl(args)

main()
