import xarray as xr
import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

BASE_PATH_DATA = 'data/skogsstyrelsen/'
IMG_PATHS_TRAIN = 'skogs_names_train.npy'
IMG_PATHS_VAL = 'skogs_names_val.npy'
IMG_PATHS_TEST = 'skogs_names_test.npy'
LABEL_PATH_TRAIN = 'skogs_gts_train.npy'
LABEL_PATH_VAL = 'skogs_gts_val.npy'
LABEL_PATH_TEST = 'skogs_gts_test.npy'
BAND_NAMES = ['b01', 'b02', 'b03', 'b04', 'b05', 'b06', 'b07', 'b08', 'b8a', 'b09', 'b11', 'b12']

# stds and means are calculated from train dataset
STDS = [0.10581802576780319, 0.09856127202510834, 0.0969739779829979, 0.09902837127447128, 0.0968807265162468, 0.10136120766401291, 0.10655524581670761, 0.10657545924186707, 0.10631011426448822, 0.1114731952548027, 0.09345966577529907, 0.10024210810661316]
MEANS = [0.095946565, 0.10766203, 0.13173726, 0.15344737, 0.19434468, 0.25526184, 0.2828849, 0.3022465, 0.31196824, 0.3159495, 0.32635692, 0.24979565]

def load_image(path):
    img = xr.open_dataset(path)
    yy_mm_dd = getattr(img, 'time').values[0]
    yy = yy_mm_dd.astype('datetime64[Y]').astype(int) + 1970
    mm = yy_mm_dd.astype('datetime64[M]').astype(int) % 12 + 1

    band_list = []
    for band in BAND_NAMES:
        if yy >= 2022 and mm >= 1: # New normalization after Jan 2022
            band_list.append((getattr(img, band).values - 1000) / 10000)
        else:
            band_list.append(getattr(img, band).values / 10000) 
            
    img = np.concatenate(band_list, axis = 0)
    img = np.transpose(img, [1,2,0])
    img = np.fliplr(img).copy()
    img = np.flipud(img).copy()

    H, W = img.shape[:2]
    
    # padding
    if H != 21 and W != 21:
        zeros = np.zeros((1, 20, 12))
        img = np.concatenate((img, zeros), axis = 0)
        zeros = np.zeros((21, 1, 12))
        img = np.concatenate((img, zeros[:]), axis = 1)
        
    elif H != 21:
        zeros = np.zeros((1, 21, 12))
        img = np.concatenate((img, zeros), axis = 0)
        
    elif W != 21:
        zeros = np.zeros((21, 1, 12))
        img = np.concatenate((img, zeros[:]), axis = 1)
        
    return img

class CustomImageDataset(Dataset):
    def __init__(self, label_dir, img_dir, transform=None, target_transform=None, use_selected_bands = False):
        self.img_labels = list(np.load(label_dir))
        self.img_dir = img_dir
        image_paths = list(np.load(img_dir))
        self.image_paths = [path[1:] for path in image_paths]
        self.transform = transform
        self.target_transform = target_transform
        self.use_selected_bands = use_selected_bands

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        image = load_image(self.image_paths[idx])
        label = self.img_labels[idx]
        
        # convert to float32
        image = np.float32(image)
        label = np.float32(label)
        
        if self.use_selected_bands:
            # dropping those bands: {'b8a', 'b07', 'b08', 'b03', 'b04', 'b05'}
            image = image[:, :, [0,1,5,9,10,11]]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# train model function
def train_model(model, criterion, optimizer, train_loader, val_loader, scheduler=None, num_epochs = 10, show_plot = True, show_log=False, sigmoid=False, cnn = False):
    
    min_loss = 10000
    
    # Track loss
    training_loss, validation_loss = [], []
    
    # Track accuracy
    training_acc, validation_acc = [], []
    
    # Track acc for clear and cloudy for training
    cloudy_acc, clear_acc = [], []
    
    for i in range(num_epochs):
        # Track loss
        epoch_training_loss, epoch_validation_loss = 0, 0
        train_size, val_size = 0, 0
        
        # track accuracy
        train_correct, val_correct = 0, 0
        cloudy_correct, clear_correct = 0, 0
        total_cloudy, total_clear = 0, 0

        # training
        model.train(True)
        for batch_nr, (data, labels) in enumerate(train_loader):
            
            if not cnn:
                data = data.view(-1,21*21*12)

            # predict
            pred = model(data)
            
            # calculate accuracy
            if sigmoid:
                pred = pred.view(-1)
                preds = torch.round(pred)
            else:
                _,preds = torch.max(pred,dim=1)

            # get correct predicted instances
            train_correct += torch.sum(preds==labels).item()
            
            # Clear stored gradient values
            optimizer.zero_grad()
            
            if not sigmoid:
                labels = labels.type(torch.int64)
                
            loss = criterion(pred, labels)
            
            # Backpropagate the loss through the network to find the gradients of all parameters
            loss.backward()
            
            # Update the parameters along their gradients
            optimizer.step()
            
            if scheduler != None:
                scheduler.step()
            
            # Update loss
            epoch_training_loss += loss.detach().numpy()
            train_size += len(data)
            
        # validation
        model.eval()
        for batch_nr, (data, labels) in enumerate(val_loader):
            
            if not cnn:
                data = data.view(-1,21*21*12)
            
            # predict
            pred = model(data)
            
            # calculate accuracy
            if sigmoid:
                pred = pred.view(-1)
                preds = torch.round(pred)
            else:
                _,preds = torch.max(pred,dim=1)
            
            val_correct += torch.sum(preds==labels).item()
            for j,p in enumerate(preds):
                if p == labels[j] and p == 0:
                    clear_correct += 1
                    total_clear += 1
                elif p == labels[j] and p == 1:
                    cloudy_correct += 1
                    total_cloudy += 1
                elif labels[j] == 0:
                    total_clear += 1
                elif labels[j] == 1:
                    total_cloudy += 1
             
            if not sigmoid:
                labels = labels.type(torch.int64)
               
            # calculate loss 
            loss = criterion(pred, labels)
            
            # check if loss is smaller than before, if so safe model
            if loss<min_loss:
                if cnn:
                    torch.save(model, 'best_cnn_model.pt')
                else:
                    torch.save(model, 'best_model.pt')
                min_loss = loss
            
            # Update loss
            epoch_validation_loss += loss.detach().numpy()
            val_size += len(data)
        
        # Save loss for plot
        training_loss.append(epoch_training_loss/train_size)
        validation_loss.append(epoch_validation_loss/val_size)
        
        # Save accuracy for plot
        training_acc.append(train_correct/train_size)
        validation_acc.append(val_correct/val_size)
        
        # Save accuracy for clear and cloudy
        clear_acc.append(clear_correct/total_clear)
        cloudy_acc.append(cloudy_correct/total_cloudy)

        if show_log:
            print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')
            print(f'Train accuracy = {train_correct/train_size}')
            print(f'Validation accuracy = {val_correct/val_size}')
            #print(f'Cloudy accuracy = {cloudy_correct/total_cloudy}')
            #print(f'Clear accuracy = {clear_correct/total_clear}')
            #print(f'number of cloudy images = {total_cloudy}')
            #print(f'number of clear images = {total_clear}')
        
    if show_plot:
        # Plot training and validation loss
        epoch = np.arange(len(training_loss))
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(epoch, training_loss, 'r', label='Training loss',)
        plt.plot(epoch, validation_loss, 'b', label='Validation loss')
        plt.legend()
        plt.xlabel('Epoch'), plt.ylabel('Loss')
        plt.show()
        
        # Plot training and validation accuracy
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(epoch, training_acc, 'r', label='Training accuracy',)
        plt.plot(epoch, validation_acc, 'b', label='Validation accuracy')
        plt.legend()
        plt.xlabel('Epoch'), plt.ylabel('Accuracy')
        plt.show()
        
        # Plot clear and cloudy accuracy
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(epoch, cloudy_acc, 'r', label='Cloudy accuracy',)
        plt.plot(epoch, clear_acc, 'b', label='Clear accuracy')
        plt.legend()
        plt.xlabel('Epoch'), plt.ylabel('Accuracy')
        plt.show()
    
    
    idx = np.argmin(validation_loss)
    print(f'lowest loss for validation set: {np.min(validation_loss)}, with an accuracy of {validation_acc[idx]}, cloud acc {cloudy_acc[idx]}, clear acc = {clear_acc[idx]}')


def test_model(model, test_loader, sigmoid=False, cnn = False):
    
    total_correct, clear_correct, cloudy_correct = 0,0,0
    total_cloudy, total_clear = 0,0
    test_labels = []
    predictions = []
    
    with torch.no_grad():
        for batch_nr, (data, labels) in enumerate(test_loader):
            
            if not cnn:
                data = data.view(-1,21*21*12)
                
            # predict
            pred = model(data)
        
             # calculate accuracy
            if sigmoid:
                pred = pred.view(-1)
                preds = torch.round(pred)
            else:
                _,preds = torch.max(pred,dim=1)
            
            total_correct += torch.sum(preds==labels).item()
            for j,p in enumerate(preds):
                if p == labels[j] and p == 0:
                    clear_correct += 1
                    total_clear += 1
                elif p == labels[j] and p == 1:
                    cloudy_correct += 1
                    total_cloudy += 1
                elif labels[j] == 0:
                    total_clear += 1
                elif labels[j] == 1:
                    total_cloudy += 1
                    
            test_labels.extend(labels)
            predictions.extend(preds)

    print("Final accuracy: %.2f%%" % (100*total_correct/(total_cloudy+total_clear)))
    print(f'Correct {total_correct} times out of {(total_cloudy+total_clear)}')

    print(f'Correct Clear {clear_correct} times out of {total_clear}: {100*clear_correct/total_clear:.2f}%')
    print(f'Correct Cloudy {cloudy_correct} times out of {total_cloudy}: {100*cloudy_correct/total_cloudy:.2f}%')
    
    print(f'Recall Score = {recall_score(test_labels, predictions)}')
    print(f'Precision Score = {precision_score(test_labels, predictions)}')
    print(f'F1 Score = {f1_score(test_labels, predictions)}')

    cm = confusion_matrix(test_labels, predictions)
    ConfusionMatrixDisplay(confusion_matrix = cm,  display_labels=['Clear', 'Cloudy']).plot()