import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from vis_and_data_utils.labels import labels
from collections import OrderedDict

id2train_id = {label.id: label.trainId for label in labels}

train_id2color = {label.trainId: label.color for label in labels }
name2train_id = {label.name: label.trainId for label in labels }

switched_labels2color = {'road_2': (84, 86, 22), 
                         'person_2': (167, 242, 242), 
                         'vegetation_2': (242, 160, 19), 
                         'car_2': (30, 193, 252), 
                         'sidewalk_2': (46, 247, 180)}

label_switches = OrderedDict([('sidewalk', 8./17.), ('person', 7./17.), ('car', 6./17.), ('vegetation', 5./17.), ('road', 4./17.)])

switched_id2name = {19+i:list(switched_labels2color.keys())[i] for i in range(len(switched_labels2color))}
switched_name2train_id = {list(switched_labels2color.keys())[i]:19+i for i in range(len(switched_labels2color))}

def get_color_map():
    
    color_map = np.zeros((256, 3), dtype = np.uint8)
    color_map[255] = [0,0,0]
    
    for train_id, color in train_id2color.items():
        color_map[train_id] =color
    for key in switched_labels2color.keys():
        color_map[switched_name2train_id[key]] = switched_labels2color[key]
    return color_map

def get_switchable_ids():
    switch_from_names = label_switches.keys()
    switch_from_ids = [name2train_id[name] for name in switch_from_names]
    switch_to_ids = [switched_name2train_id[name + "_2"] for name in switch_from_names]
    return switch_from_ids + switch_to_ids

def seg_to_rgb(segm, id_to_color):
    return id_to_color[segm.squeeze().int().numpy()]

def show_dataset_random_examples(dataset, n=4):
    plt.figure(figsize=(20,10))
    random_pos = int(np.random.random()*(len(dataset) - n))
    img = np.hstack((dataset[random_pos + j][0].permute(1,2,0)
                    for j in range(n)))
    plt.imshow(img)
    plt.axis('off')
    
def show_item(dataset, ind = None):
    id_to_color = get_color_map()
    if ind == None:
        ind = int(np.random.random()*len(dataset))
    fig, axs = plt.subplots(1, 2, figsize=(15, 4), constrained_layout=True)
    im, target = dataset[ind]
    axs[0].imshow(im.permute(1,2,0))
    axs[0].axis('off')
    axs[1].imshow(seg_to_rgb(target, id_to_color))
    axs[1].axis('off')
    plt.show()

def plot_test_batch_with_results(batch_imgs, batch_segms, results, outdir = None):
    batch_size = batch_imgs.shape[0]
    
    cmap = get_color_map()
    rows = 3

    fig = plt.figure(figsize=(batch_size * 4, rows * 2))
    gs = gridspec.GridSpec(rows, batch_size, wspace=0.0, hspace=0.0)
    
    for img_num in range(batch_size):
        ax = plt.subplot(gs[0,img_num])
        ax.axis('off')
        ax.imshow(batch_imgs[img_num].permute(1,2,0))
        
        ax = plt.subplot(gs[1,img_num])
        ax.axis('off')
        ax.imshow(seg_to_rgb(batch_segms[img_num], cmap))

        ax = plt.subplot(gs[2,img_num])
        ax.axis('off')
        ax.imshow(seg_to_rgb(results[img_num], cmap))
        
    fig.tight_layout()
    if outdir:
        fig.savefig(outdir, dpi=200, bbox_inches="tight", pad_inches=0)
        
        
def plot_batch_with_results(batch_imgs, batch_segms, results, outdir = None):
    """
    Images, gt segmentations and generated segmentations
    Input:
        batch_imgs - tensor of size (img_num, 3, h, w)
        batch_segms - tensor of size (img_num, m, h, w)
        results - tensor of size (img_num, n, h, w)
    Output:
        
    """
    num_predictions = results.shape[1]
    num_gt = batch_segms.shape[1]
    batch_size = batch_imgs.shape[0]
    
    cmap = get_color_map()
    
    rows = 1 + num_gt + num_predictions

    fig = plt.figure(figsize=(batch_size * 4, rows * 2))
    gs = gridspec.GridSpec(rows, batch_size, wspace=0.0, hspace=0.0)
    
    for img_num in range(batch_size):
        ax = plt.subplot(gs[0,img_num])
        ax.axis('off')
        ax.imshow(batch_imgs[img_num].permute(1,2,0))
        
        for seg_ind in range(num_gt):
            ax = plt.subplot(gs[seg_ind + 1,img_num])
            ax.axis('off')
            ax.imshow(seg_to_rgb(batch_segms[img_num][seg_ind], cmap))

        for pred_ind in range(num_predictions):
            ax = plt.subplot(gs[pred_ind + num_gt + 1,img_num])
            ax.axis('off')
            ax.imshow(seg_to_rgb(results[img_num][pred_ind], cmap))
    fig.tight_layout()
    if outdir:
        fig.savefig(outdir, dpi=200, bbox_inches="tight", pad_inches=0)