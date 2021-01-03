from __future__ import print_function

# std libs
import os
import sys 
import copy
import time
import random
import argparse
import numpy as np
from math import log10

#AutoClassify
from AutoClassify import Classifier, Scanpy_IO, CSV_IO

# reading in single cell data using scanpy
import scanpy as sc


# torch libs
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.nn.functional as F
## in a future release vv
from tensorboardX import SummaryWriter

# anamoly detection
torch.autograd.set_detect_anomaly(True)

# lr = 0.0002
lr = 0.0002


parser = argparse.ArgumentParser()

# classifier options
parser.add_argument('--ClassifierOnly', type=bool, default=False, help='running the classifer only, default = False')
parser.add_argument('--ClassifierEpochs', type=int, default=50, help='number of epochs to train the classifier, default = 50')

parser.add_argument("--hdim", type=int, default=128, help="dim of the latent code, Default=128")
parser.add_argument("--save_iter", type=int, default=1, help="Default=1")
parser.add_argument("--test_iter", type=int, default=1000, help="Default=1000")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=24)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')

parser.add_argument("--nEpochs", type=int, default=200, help="number of training epochs, default = 200 ")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--lr', type=float, default=lr, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument('--clip', type=float, default=100, help='the threshod for clipping gradient')
parser.add_argument("--step", type=int, default=500, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument('--cuda', default = True ,action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--tensorboard', default=True ,action='store_true', help='enables tensorboard, default True')
parser.add_argument('--outf', default='./TensorBoard/', help='folder to output training stats for tensorboard')
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")


def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)

str_to_list = lambda x: [int(xi) for xi in x.split(',')]



def main():
    global opt, model
    opt = parser.parse_args()
#     print(opt);
    
    
    """
    
    Loading in Dataset: We will load in the normalized count matrix from Tabula Muris data
    ## note: we have normalized the data already and saved as scanpy annotated data format
    
    
    """
    ## FOR NOW WE WILL MANUALLY PICK WHICH OPTIONS WE WANT FOR IO
    
    print("==> Reading in Data")
#     # if we have h5ad from a scanpy or seurat object 
#     train_data_loader, valid_data_loader = Scanpy_IO('/home/jovyan/68K_PBMC_scGAN_Process/raw_68kPBMCs.h5ad',
#                                                     batchSize=opt.batchSize, 
#                                                     workers = opt.workers)
#         # get the input size
#     inp_size = [batch[0].shape[1] for _, batch in enumerate(valid_data_loader, 0)][0];
    
    # if we have CSV turned to h5 (pandas dataframe)
    train_set, test_set = CSV_IO("/home/jovyan/ACTINN/train.h5","/home/jovyan/ACTINN/train_lab.csv", "/home/jovyan/ACTINN/test.h5")

    sys.exit("COMPLETED READING")
    
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=opt.outf)
    
    start_time = time.time()
    cur_iter = 0
    
    
    """ 
    
    Building the classifier model:
    
    """
    cf_model = Classifier(output_dim = 11, input_size = inp_size).cuda()
    cf_criterion = torch.nn.CrossEntropyLoss()
    cf_optimizer = torch.optim.Adam(params=cf_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005, amsgrad=False)
    cf_decayRate = 0.95
    cf_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=cf_optimizer, gamma=cf_decayRate)
    print("\n Classifier Model \n")
    print(cf_model)
    
    """
    
    Training as the classifier (Should be done when we are warm-starting the VAE part)
    
    """
    
    def train_classifier(cf_epoch, iteration, batch, cur_iter):  
        
        if len(batch[0].size()) == 3:
            batch = batch[0].unsqueeze(0)
        else:
            labels = batch[1]
            batch = batch[0]
        batch_size = batch.size(0)
                       
        features= Variable(batch).cuda()
        true_labels = Variable(labels).cuda()
                
        info = f"\n====> Classifier Cur_iter: [{cur_iter}]: Epoch[{cf_epoch}]({iteration}/{len(train_data_loader)}): time: {time.time()-start_time:4.4f}: "
        
        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'
            
        # =========== Update the classifier ================                  
        pred_cluster = cf_model(features) 
        
        loss = cf_criterion(pred_cluster.squeeze(), true_labels)
        cf_optimizer.zero_grad()      
        loss.backward()                   
        cf_lr_scheduler.step() 
     
        info += f'Cross Entropy Loss: {loss.data.item():.4f} '     
        print(info)
        
        
        
    # TRAIN     
    for epoch in range(0, opt.ClassifierEpochs + 1): 
        #save models
        if epoch % 5 == 0 :
            save_epoch = (epoch//opt.save_iter)*opt.save_iter   
            save_checkpoint_classifier(cf_model, save_epoch, 0, '')

        cf_model.train()
    for iteration, batch in enumerate(train_data_loader, 0):
            #--------------train Classifier Only------------
            train_classifier(epoch, iteration, batch, cur_iter);
            cur_iter += 1
        
    print(f"TOTAL TRAINING TIME {time.time() - start_time}"); 
        
    

            
def load_model(model, pretrained):
        weights = torch.load(pretrained)
        pretrained_dict = weights['Saved_Model'].state_dict()  
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)

def save_checkpoint_classifier(model, epoch, iteration, prefix=""):
        dir_path = "./ClassifierWeights/"
        model_out_path = dir_path + prefix +"model_epoch_{}_iter_{}.pth".format(epoch, iteration)
        state = {"epoch": epoch ,"Saved_Model": model}
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        torch.save(state, model_out_path)

        print("Classifier Checkpoint saved to {}".format(model_out_path))
        
def evaluate_classifier(valid_data_loader, cf_model):
        print("==> Evaluating on Validation Set:")
        cf_model.eval()
        for iteration, batch in enumerate(valid_data_loader,0):
            y_pred = cf_model(batch[0].cuda())
            after_train = cf_criterion(y_pred.squeeze(),batch[1].cuda()) 
            print(f'    ->Validation Cross Entropy Loss: {after_train.item()}')

    
if __name__ == "__main__":
    main()   
    
    
    