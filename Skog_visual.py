import numpy as np
import netCDF4 as nc
import json
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os
from sklearn.metrics import roc_curve, roc_auc_score

BASE_PATH_DATA = '../data/skogsstyrelsen/'
img_paths_train = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_train.npy')))
img_paths_val = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_val.npy')))
img_paths_test = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_test.npy')))
json_content_train = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_train.npy'), allow_pickle=True))
json_content_val = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_val.npy'), allow_pickle=True))

BAND_NAMES = ['b01', 'b02', 'b03', 'b04', 'b05', 'b06', 'b07', 'b08', 'b8a', 'b09', 'b11', 'b12']

# stds and means are calculated from train dataset
stds = [0.10581802576780319, 0.09856127202510834, 0.0969739779829979, 0.09902837127447128, 0.0968807265162468, 0.10136120766401291, 0.10655524581670761, 0.10657545924186707, 0.10631011426448822, 0.1114731952548027, 0.09345966577529907, 0.10024210810661316]
means = [0.095946565, 0.10766203, 0.13173726, 0.15344737, 0.19434468, 0.25526184, 0.2828849, 0.3022465, 0.31196824, 0.3159495, 0.32635692, 0.24979565]

class Preprocess:
    
    def __init__(self, name, img_path):
        self.name = name
        self.img_path = img_path
        
    def store_band(self, img_index):
        
        self.img = xr.open_dataset(self.img_path[img_index])
        yy_mm_dd = getattr(self.img, 'time').values[0]
        yy = yy_mm_dd.astype('datetime64[Y]').astype(int) + 1970
        mm = yy_mm_dd.astype('datetime64[M]').astype(int) % 12 + 1
        
        self.band_list = []
        self.band_dict = {}
        for band in BAND_NAMES:
            if yy >= 2022 and mm >= 1: # New normalization after Jan 2022
                band_value = (getattr(self.img, band).values - 1000) / 10000
                self.band_dict[band] = band_value
                self.band_list.append(band_value)
            else:
                band_value = getattr(self.img, band).values / 10000
                self.band_dict[band] = band_value
                self.band_list.append(band_value)
        return self.band_list, self.band_dict
    
    def band_df(self):
        """
        For understanding features, this method store all bands values 
        of the img_path in a pandas dataframe, column is band_names. 
        """
        self.df = self.store_band(0)[1]
        for band in BAND_NAMES:
            for i in range(1,len(self.img_path)):
                self.df[band] = np.concatenate((
                    self.store_band(i)[1][band],
                    self.df[band]), axis = None)
            self.df[band] = np.squeeze(self.df[band])
            
        self.df = pd.DataFrame(self.df)
        # Save DataFrame to pickle file
        self.df.to_pickle(f"{self.name} bands dataframe.pkl")
        return self.df  
    
    def compose_img(self, img_index):
        """
        shape image to H x W x band_channel 
        """

        self.img = np.concatenate(self.store_band(img_index)[0], axis=0)
        self.img = np.transpose(self.img, [1,2,0])
        self.img = np.fliplr(self.img).copy()
        self.img = np.flipud(self.img).copy()
        
        return self.img
    
    def normal_img(self, img_index):
        """
        shape image to H x W x band_channel with normalized bands value
        """

        img = self.compose_img(img_index)
        H, W = img.shape[:2]
        img = np.reshape((img - means) / stds, [H, W, len(BAND_NAMES)])
        return img
    
    def show_rgb(self, img_index):
        """
        show RGB image for SFA dataset
        Parameters
        ----------
        img_path: list (e.g.img_paths_train, img_paths_val)
        img_index : int

        """
        img = self.compose_img(img_index)
        self.rgb_img = img[:, :, [3,2,1]]/np.max(img[:, :, [3,2,1]])
        plt.imshow(self.rgb_img, vmin=0, vmax=1)
        
    def scatter_band(self):
        # scatter plot bands
        try:
            df = pd.read_pickle(f"{self.name} bands dataframe.pkl")
        except FileNotFoundError:
            df = self.band_df()
            
        for band_1 in list(BAND_NAMES):
            for band_2 in list(BAND_NAMES):            
                if band_1 != band_2:
                    scatter = sn.scatterplot(
                        x=df[band_1],
                        y=df[band_2])
                    scatter.set_title(f"{band_1} vs {band_2}")
                    # sc = scatter.get_figure()
                    # sc.savefig(f"{band_1} vs {band_2}", dpi = 300)
                    plt.show(scatter)
                    
    def corr_heatmap(self):
        try:
            df = pd.read_pickle(f"{self.name} bands dataframe.pkl")
        except FileNotFoundError:
            df = self.band_df()
            
        corr_matrix = df.corr()
        plt.figure(figsize=(20,15))
        heatmap = sn.heatmap(corr_matrix, annot=True)
        hm = heatmap.get_figure()
        hm.savefig("correlation_heatmap.png", dpi = 300)
        plt.show()
        return df

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
    
def stds_means(df):
    band_std = []
    band_mean = []
    for band in BAND_NAMES:
        band_std.append(df[band].std())
        band_mean.append(df[band].mean())

    return band_std, band_mean


def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

        # If correlation exceeds the threshold
        if val >= threshold:
            # Print the correlated features and the correlation value
            #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
            drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    print('Removed Columns {}'.format(drops))
    return x

# stds, means = stds_means(df)    
skog_train = Preprocess("skog train",img_paths_train)
df = skog_train.corr_heatmap()
# remove highly correlated bands, set threshold 0.97
remove_collinear_features(df, 0.97)