import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from labels import labels 

def get_color_map(labels = labels):
    """from https://github.com/tensorflow/models/blob/master/research/deeplab/utils/get_dataset_colormap.py """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap
    #id_to_color = np.zeros((256, 3), int)
    #for item in labels:
    #    if item.trainId > 0:
    #        id_to_color[item.trainId] = item.color
    #return id_to_color

def seg_to_rgb(segm, id_to_color):
    return id_to_color[segm.permute(1,2,0).squeeze().int().numpy()]

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