from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tqdm import trange

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F


from model.model import ProbUNet
import vis_and_data_utils.visualization_utils as visualization_utils
from vis_and_data_utils.labels import labels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 24
m = 3

def get_id_to_train_id(labels = labels):
    id_to_train_id = np.zeros((len(labels), 1), int)
    for item in labels:
        if item.id > 0:
            if item.trainId == 255:
                id_to_train_id[item.id] = 0
            else:
                id_to_train_id[item.id] = item.trainId 
    return id_to_train_id

def get_train_id_to_id(labels = labels):
    train_id_to_id = np.zeros((len(labels), 1), int)
    for item in labels:
        if item.trainId > 0:
            train_id_to_id[item.trainId] = item.id
    return train_id_to_id

def get_segmentation_variant(segm, classes_flip_vec = None):
    
    switch_from_names = visualization_utils.label_switches.keys()
    switch_from_ids = [visualization_utils.name2train_id[name] for name in switch_from_names]
    switch_to_ids = [visualization_utils.switched_name2train_id[name + "_2"] for name in switch_from_names]
    
    if classes_flip_vec is None:
        class_change_probabilities = list(visualization_utils.label_switches.values())
        classes_flip_vec = (np.random.random_sample(5) < class_change_probabilities)
    
    res = segm.clone()
    for label_ind in range(5):
        if classes_flip_vec[label_ind]:
            res[res == switch_from_ids[label_ind]] = switch_to_ids[label_ind]
    return res

def create_possible_segm_with_probs(segm):
    """
        segm - torch tensor (h,w)
    """
    res = segm.repeat((32, 1, 1))
    probs = torch.ones(32)
    
    switch_from_names = visualization_utils.label_switches.keys()
    switch_from_ids = [visualization_utils.name2train_id[name] for name in switch_from_names]
    switch_to_ids = [visualization_utils.switched_name2train_id[name + "_2"] for name in switch_from_names]
    switch_probs = list(visualization_utils.label_switches.values())
    
    for segm_ind, segm_bool_mask in enumerate(product([True,False], repeat=5)):
        for label_ind in range(len(segm_bool_mask)):
            if segm_bool_mask[label_ind]:
                res[segm_ind][res[segm_ind] == switch_from_ids[label_ind]] = switch_to_ids[label_ind]
                probs[segm_ind] *= switch_probs[label_ind]
            else:
                probs[segm_ind] *= (1. - switch_probs[label_ind])
    return res, probs

def train_epoch(model, optimizer, train_loader):
    loss_log = []
    model.train()
    for _, (x_batch, y_batch) in zip(trange(len(train_loader)), train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        model_loss = model.compute_lower_bound(x_batch, y_batch)
        model_loss.backward()
        #print("Grad: ", model.unet.input_block.conv_net[0].weight.grad)
        optimizer.step()
        loss = model_loss.item()
        loss_log.append(loss)
    return loss_log

def test(model, epoch_num, test_loader, res_dir = None):
    ce_log = []
    ged_log = []
    model.eval()
    for batch_num, (x_batch, y_batch) in zip(trange(len(test_loader)), test_loader):  
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        if res_dir and batch_num == 0:
            output = model.sample_m(x_batch, m)
            loss = 0
            for ind in range(m):
                loss += nn.CrossEntropyLoss()(output[:,ind], y_batch.squeeze()).item()
            loss = loss/m
            ce_log.append(loss)
            pred = torch.argmax(output, dim = 2)
            ged = 0
            for i in range(test_loader.batch_size):
                ged += get_energy_distance(pred[i].squeeze(), y_batch[i].squeeze())
            ged_log.append((ged/test_loader.batch_size).numpy())
            visualization_utils.plot_batch_with_results(x_batch.cpu(), 
                                                        y_batch.cpu(), 
                                                        pred.cpu().squeeze(), 
                                                        outdir = res_dir + "epoch_{}.pdf".format(epoch_num))
            
        else:
            output = model.sample(x_batch)
            loss = nn.CrossEntropyLoss()(output, y_batch.squeeze())
            ce_log.append(loss.item())


    return ce_log, ged_log

#def plot_history(train_history, ce_log, GED_log, epoch_num, save_dir = "results/"):
#    plt.figure()
#    plt.title('Results for epoch {}'.format(epoch_num))
#    plt.plot(train_history, label='train', zorder=1)
    
#    points = np.array(ce_log)
#    plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='val_ce', zorder=2)
#    points = np.array(acc_log)
#    plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='val_acc', zorder=2)
        
#    plt.xlabel('train steps')

#    plt.legend(loc='best')
#    plt.grid()
    
#    plt.savefig(save_dir + "loss_"+str(epoch_num) + ".png")
    
def plot_history(train_history, epoch_num, title='loss', save_dir = "results/"):
    plt.figure()
    plt.title('Results of {} for epoch {}'.format(title, epoch_num))
    plt.plot(train_history, zorder=1)
    
    plt.xlabel('train steps')
    
    #plt.legend(loc='best')
    plt.grid()

    plt.savefig(save_dir + title + "_"+str(epoch_num) + ".png")
    
    
def plot_val_history(train_history, epoch_num, title='loss', save_dir = "results/"):
    plt.figure()
    plt.title('Results of {} for epoch {}'.format(title, epoch_num))
    
    plt.xlabel('train steps')
    
    points = np.array(train_history)
    plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='val_acc', zorder=2)
    plt.grid()

    plt.savefig(save_dir + title + "_"+str(epoch_num) + ".png")
    
def train(model, opt, n_epochs, train_loader, test_loader, save_path = None):
    train_log = []
    ce_log, GED_log = [], []
    steps = len(train_loader)

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, opt, train_loader)
        train_log.extend(train_loss)
        plot_history(train_log, epoch, title = "loss")  
        
        val_loss, val_ged = test(model, epoch, test_loader, res_dir = "results/")
        ce_log.append((steps * (epoch + 1), np.mean(val_loss)))
        GED_log.append((steps * (epoch + 1), np.mean(val_ged)))
        plot_val_history(ce_log, epoch, title = "cross_entropy") 
        plot_val_history(GED_log, epoch, title = "GED") 
        
        if save_path:
            torch.save(model.state_dict(), save_path)
            
    #print("Final error: {:.2%}".format(1 - acc_log[-1][1]))
    
def compute_iou(segm_1, segm_2, classes, loss_mask=None):
    """
        Computes IoU metrics for two segmentations only for given classes
            (if class is not presented on both images, it is not considered in mean)
        Input:
            segm_i shape is (h,w), torch tensor
            loss_mask = (h,w), bool, 1 what to count, torch tensor
    """

    if loss_mask is None:
        loss_mask = torch.ones(segm_1.shape[0], segm_1.shape[1], dtype = torch.uint8).cuda()
        
    iou = torch.zeros(1,dtype=torch.float32)
    
    considered_classes = 0

    for i,c in enumerate(classes):

        pred_ = (segm_1 == c)
        labels_ = (segm_2 == c)

        TP = (pred_*labels_*loss_mask).sum()
        FP = (pred_ * (~labels_) * loss_mask).sum()
        FN = ((~pred_) * labels_  * loss_mask).sum()

        if TP + FP + FN != 0:
            iou += TP/(TP + FP + FN)
            considered_classes += 1

    return iou/considered_classes

def get_energy_distance(S, gt):
    """
        Computes energy distance metric
        Input:
            S - segmentation variants (torch tensor of size (variants_num, h,w)) 
            gt - ground truth segmentation (torch tensor of size  (h,w))
        Output:
            D_ed - energy_distance value (float)
    """
    n = len(S)
    classes_list = visualization_utils.get_switchable_ids()
    
    Y, probs = create_possible_segm_with_probs(gt)
    
    SY = 0
    SS = 0
    YY = 0
    
    for S_i in S:
        for j, Y_j in enumerate(Y):
            SY += (1 - compute_iou(S_i, Y_j, classes_list))*probs[j]
    
    SY = SY*2/n
    
    for S_i in S:
        for S_j in S:
            SS += (1 - compute_iou(S_i, S_j, classes_list))
            
    SS /= n**2
    
    for i,Y_i in enumerate(Y):
        for j,Y_j in enumerate(Y):
            YY += (1 - compute_iou(Y_i, Y_j, classes_list))*probs[j]*probs[i]

    return SY - SS - YY