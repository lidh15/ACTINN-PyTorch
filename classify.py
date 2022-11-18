from __future__ import print_function

# std libs
import time
from tqdm import tqdm
import argparse
import numpy as np

#AutoClassify
from ACTINN import Classifier, CSV_IO
from ACTINN.utils import evaluate_classifier, init_weights, load_model, save_checkpoint_classifier

# torch libs
import torch
import torch.utils.data
import torch.nn.parallel
from torch.autograd import Variable

import torch.nn.functional as F

# anamoly detection
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()

# classifier options
parser.add_argument(
    '--ClassifierEpochs',
    type=int,
    default=50,
    help='number of epochs to train the classifier, default = 50')

parser.add_argument('--data_type',
                    type=str,
                    default="scanpy",
                    help='type of train/test data, default="scanpy"')
parser.add_argument("--save_iter", type=int, default=1, help="Default=1")
parser.add_argument('--workers',
                    type=int,
                    help='number of data loading workers',
                    default=24)
parser.add_argument('--batchSize',
                    type=int,
                    default=128,
                    help='input batch size')
parser.add_argument("--start_epoch",
                    default=1,
                    type=int,
                    help="Manual epoch number (useful on restarts)")
parser.add_argument('--print_frequency',
                    type=int,
                    default=5,
                    help='frequency of training stats printing, default=5')
parser.add_argument('--lr',
                    type=float,
                    default=0.0001,
                    help='learning rate, default=0.0001')
parser.add_argument("--momentum",
                    default=0.9,
                    type=float,
                    help="Momentum, Default: 0.9")
parser.add_argument('--clip',
                    type=float,
                    default=100,
                    help='the threshod for clipping gradient')
parser.add_argument(
    "--step",
    type=int,
    default=1000,
    help=
    "Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=1000"
)
parser.add_argument('--cuda',
                    default=True,
                    action='store_true',
                    help='enables cuda, default = True')
parser.add_argument('--manualSeed',
                    type=int,
                    default=0,
                    help='manual seed, default = 0')
parser.add_argument('--tensorboard',
                    default=True,
                    action='store_true',
                    help='enables tensorboard, default True')
parser.add_argument('--outf',
                    default='./TensorBoard/',
                    help='folder to output training stats for tensorboard')
parser.add_argument("--pretrained",
                    default="",
                    type=str,
                    help="path to pretrained model (default: none)")
parser.add_argument("--data_prefix",
                    default="workspace",
                    type=str,
                    help="prefix to data path (default: workspace)")
parser.add_argument("--predict",
                    default="",
                    type=str,
                    help="predict data path")
parser.add_argument("--result",
                    default="",
                    type=str,
                    help="predict result path")
parser.add_argument("--model_dir",
                    default="",
                    type=str,
                    help="model save path")
parser.add_argument("--result_header",
                    default="cell type",
                    type=str,
                    help="result column name")
parser.add_argument("--classes_number",
                    default=10,
                    type=int,
                    help="number of classes")


def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)


str_to_list = lambda x: [int(xi) for xi in x.split(',')]


def main():
    global opt, model
    opt = parser.parse_args()

    # determin the device for torch
    ## if we are allowed to run things on CUDA
    if opt.cuda and torch.cuda.is_available():
        device = "cuda"
        print('==> Using GPU (CUDA)')

    else:
        device = "cpu"
        print('==> Using CPU')
        print(
            '    -> Warning: Using CPUs will yield to slower training time than GPUs'
        )
    """
    
    Loading in Dataset: We will load in the normalized count matrix from Tabula Muris data
    ## note: we have normalized the data already and saved as scanpy annotated data format
    
    
    """
    if opt.data_type.lower() == "csv":
        data_prefix = opt.data_prefix
        if opt.predict:
            cf_model = torch.load(opt.pretrained)["Saved_Model"]
            cf_model.eval()
            _, valid_data_loader, _, _ = CSV_IO(
                '',
                '',
                opt.predict,
                '',
                batchSize=opt.batchSize,
                workers=opt.workers,
                sum_index=cf_model.sum_index,
                train_index=cf_model.train_index)
            prediction = evaluate_classifier(valid_data_loader,
                                             cf_model,
                                             predict_only=True)
            result = opt.result if opt.result else f"result-{datetime.now():%m-%d-%H-%M-%S}.csv"
            with open(result, "w") as f:
                # header = ["Cell type\n"] ? ["cell type\n"]
                header = [f"{opt.header}\n"]
                f.writelines(header +
                             [f"{cf_model.labels[i]}\n" for i in prediction])
            print(f"\tsaving result into {result}")
            return
        else:
            # if we have CSV turned to h5 (pandas dataframe)
            train_path = f"/{data_prefix}/train.h5"
            train_lab_path = f"/{data_prefix}/train_lab.csv"

            test_path = f"/{data_prefix}/test.h5"
            test_lab_path = f"/{data_prefix}/test_lab.csv"

            train_data_loader, valid_data_loader, idx2, labels = CSV_IO(
                train_path,
                train_lab_path,
                test_path,
                test_lab_path,
                batchSize=opt.batchSize,
                workers=opt.workers)

        # get input output information for the network
        inp_size = [
            batch[0].shape[1] for _, batch in enumerate(train_data_loader, 0)
        ][0]
        # as WPPCC22 required
        number_of_classes = opt.classes_number
        # print(number_of_classes)

    else:
        raise ValueError(
            "Wrong data type, please provide Scanpy/Seurat object or h5 dataframe"
        )

    start_time = time.time()
    cur_iter = 0
    """ 
    
    Building the classifier model:
    
    """
    cf_model = Classifier(output_dim=number_of_classes,
                          input_size=inp_size).to(device)
    cf_model.set_meta(idx2, labels)
    # initilize the weights in our model
    cf_model.apply(init_weights)
    cf_criterion = torch.nn.CrossEntropyLoss()

    cf_optimizer = torch.optim.Adam(params=cf_model.parameters(),
                                    lr=0.0001,
                                    betas=(0.9, 0.999),
                                    eps=1e-08,
                                    weight_decay=0.005,
                                    amsgrad=False)
    cf_decayRate = 0.95
    cf_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=cf_optimizer, gamma=cf_decayRate)
    print("\n Classifier Model \n")
    print(cf_model)
    """
    
    Training as the classifier (Should be done when we are warm-starting the VAE part)
    
    """
    def train_classifier(cf_epoch, iteration, batch, cur_iter):

        cf_optimizer.zero_grad()
        if len(batch[0].size()) == 3:
            batch = batch[0].unsqueeze(0)
        else:
            labels = batch[1]
            batch = batch[0]
        batch_size = batch.size(0)

        features = Variable(batch).to(device)
        true_labels = Variable(labels).to(device)

        info = f"\n====> Classifier Cur_iter: [{cur_iter}]: Epoch[{cf_epoch}]({iteration}/{len(train_data_loader)}): time: {time.time()-start_time:4.4f}: "

        # =========== Update the classifier ================
        pred_cluster = cf_model(features)
        loss = cf_criterion(pred_cluster.squeeze(), true_labels)
        loss.backward()
        cf_optimizer.step()

        # decaying the LR
        if cur_iter % opt.step == 0 and cur_iter != 0:
            cf_lr_scheduler.step()
            for param_group in cf_optimizer.param_groups:
                print(f"    -> Decayed lr to -> {param_group['lr']}")

        # =========== Printing Network Information ================
        info += f'Loss: {loss.data.item():.4f} '

        if cur_iter == 0:
            print("    ->Initial stats:", info)

        if epoch % opt.print_frequency == 0 and iteration == (
                len(train_data_loader) - 1):
            print(info)

    # TRAIN
    print("---------------- ")
    print("==> Trainig Started ")
    print(f"    -> lr decaying after every {opt.step} steps")
    print(
        f"    -> Training stats printed after every {opt.print_frequency} epochs"
    )
    for epoch in tqdm(range(0, opt.ClassifierEpochs + 1),
                      desc="Classifier Training"):
        #save models
        if epoch % opt.print_frequency == 0 and epoch != 0:
            evaluate_classifier(valid_data_loader, cf_model)
            save_epoch = (epoch // opt.save_iter) * opt.save_iter

        cf_model.train()
        for iteration, batch in enumerate(train_data_loader, 0):
            #============train Classifier Only============
            train_classifier(epoch, iteration, batch, cur_iter)
            cur_iter += 1

    save_epoch = (epoch // opt.save_iter) * opt.save_iter

    save_checkpoint_classifier(cf_model, save_epoch, 0, 'LAST', opt.model_dir)
    print("==> Final evaluation on validation data: ")
    evaluate_classifier(valid_data_loader,
                        cf_model,
                        classification_report=True)
    print(f"==> Total training time {time.time() - start_time}")


if __name__ == "__main__":
    main()
