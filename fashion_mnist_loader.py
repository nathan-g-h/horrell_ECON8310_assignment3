import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

#########################
####   File paths    ####
#########################
test_images_file = "C:/Users/nghor/OneDrive/Documents/fashion_mnist/t10k-images-idx3-ubyte.gz"
test_labels_file = "C:/Users/nghor/OneDrive/Documents/fashion_mnist/t10k-labels-idx1-ubyte.gz"
model_path = "fashion_mnist_model.pt"

#################################
####    Data reader class    ####
#################################
class FashionMNISTDataset(Dataset):
    def __init__(self, images_file, labels_file):
        self.images = self.load_images(images_file)
        self.labels = self.load_labels(labels_file)

    def load_images(self, file):
        with gzip.open(file, 'rb') as f:        # open and read gz file
            
            f.read(16)                          # skip 16 btye header
            byte_data = f.read()                # read remaining bytes
            data = np.frombuffer(byte_data, dtype=np.uint8)     # and convert to array
            
            # reshape array to (number of images, height, width)
            num_images = len(data) // (28 * 28)         
            data = data.reshape(num_images, 28, 28)
            
            # normalize pixel values to values between 0 and 1
            data = data / 255.0
            
            # cnvert to torch tensor and add channel dimension (with value of 1 for grayscale)
            image_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
            
        return image_tensor

    def load_labels(self, file):
        with gzip.open(file, 'rb') as f:        # open and read gz file
            
            f.read(8)                           # skip 8 btye header          
            byte_data = f.read()                # read remaining bytes
            labels = np.frombuffer(byte_data, dtype=np.uint8)     # and convert to array
            
            # convert to torch tensor
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
        return labels_tensor

    def __len__(self):
        return len(self.labels)     # total number of images in the dataset

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]   # returns an image and its label at the given index        

#################################################
####    Initialize dataset and dataloader    ####
#################################################

test_dataset = FashionMNISTDataset(test_images_file, test_labels_file)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

##########################################
####    Create Neural Network Model   ####
##########################################
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = FashionMNISTModel()

#################################
####    Loading the model    ####
#################################

def load_model(path=model_path):
    model = FashionMNISTModel()
    optimizer = torch.optim.Adam(model.parameters())

    # load model from path
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    # set model to eval mode
    model.eval()
    print(f"Model loaded, trained for {epoch} epochs.")

    return model, optimizer, epoch

model, optimizer, trained_epoch = load_model()

###################################
####    Evaluating the model   ####
###################################
def evaluate(model, loader):
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy of the model on test images: {accuracy * 100:.2f}%")

evaluate(model, test_loader)