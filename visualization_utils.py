import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from labels import labels 

def get_color_map(labels = labels):
    id_to_color = np.zeros((256, 3), int)
    for item in labels:
        if item.trainId > 0:
            id_to_color[item.trainId] = item.color
    return id_to_color

def seg_to_rgb(segm, id_to_color):
    return id_to_color[segm.cpu().permute(1,2,0).int().squeeze().numpy()]

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
    axs[0].imshow(dataset[ind][0].permute(1,2,0))
    axs[0].axis('off')
    axs[1].imshow(seg_to_rgb(dataset[ind][1], id_to_color))
    axs[1].axis('off')
    plt.show()

def plot_batch_with_results(batch_imgs, batch_segms, results, outdir = None):
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
        ax.imshow(batch_imgs[img_num].permute(1,2,0).cpu())
        
        for seg_ind in range(num_gt):
            ax = plt.subplot(gs[seg_ind + 1,img_num])
            ax.axis('off')
            ax.imshow(seg_to_rgb(batch_segms[img_num][seg_ind], cmap))

        for pred_ind in range(num_predictions):
            ax = plt.subplot(gs[pred_ind + num_gt + 1,img_num])
            ax.axis('off')
            ax.imshow(seg_to_rgb(results[img_num][seg_ind], cmap))
    fig.tight_layout()
    if outdir:
        fig.savefig(outdir, dpi=200, bbox_inches="tight", pad_inches=0)