import numpy as np
import netCDF4 as nc
import json
import xarray as xr
import matplotlib.pyplot as plt
import os
# from sklearn.metrics import roc_curve, roc_auc_score

BASE_PATH_DATA = '../data/skogsstyrelsen/'
img_paths_train = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_train.npy')))
img_paths_val = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_val.npy')))
img_paths_test = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_test.npy')))
json_content_train = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_train.npy'), allow_pickle=True))
json_content_val = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_val.npy'), allow_pickle=True))

class SkogDataVisualization:
    """
    for inspect and understand the dataset
    """
    BAND_NAMES = ['b01', 'b02', 'b03', 'b04', 'b05', 'b06', 'b07', 'b08', 'b8a', 'b09', 'b11', 'b12']
    
    def __init__(self, name, img_path):
        self.name = name
        self.img_path = img_path
    def rgb(self, img_index):
        """
        show RGB image for SFA dataset
        Parameters
        ----------
        img_path: list (e.g.img_paths_train, img_paths_val)
        img_index : int

        """
        self._img = xr.open_dataset(self.img_path[img_index])
        self.band_list = []
        for band_name in self.BAND_NAMES:
        	self.band_list.append(getattr(self._img, band_name).values/ 10000)  # 10k division
        self._img = np.concatenate(self.band_list, axis=0)
        self._img = np.transpose(self._img, [1,2,0])
        self._img = np.fliplr(self._img).copy()
        self._img = np.flipud(self._img).copy()
        self.rgb_img = self._img[:, :, [3,2,1]]/np.max(self._img[:, :, [3,2,1]])
        plt.imshow(self.rgb_img, vmin=0, vmax=1)
        
        return self.rgb_img
    
    def heatmap(self):
        pass
        

class SkogResultVisualization:
    """
    for model performance visualization
    """
    
    def __init__(self, name, img_path):
        self.name = name
        self.img_path = img_path
    
    
    def roc_auc(self, y, y_pred):

        fpr, tpr, thresholds = roc_curve(y, y_pred)
        auc = roc_auc_score(y, y_pred)

        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--') # Plot the random guess line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve of SFA validation dataset')
        plt.legend()
        plt.show()
    
    
skog_train_2 = SkogDataVisualization("image_train_2",img_paths_train)
skog_train_2.rgb(1)