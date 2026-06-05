import os
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from PIL import Image




## The model used to generate the embeddings + its setup (i.e. freezing its internals and modifying the output layer)
embedderModel = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for parameters in embedderModel.parameters():
    parameters.requires_grad = False
embedderModel.fc = nn.Identity() # replacing the final layer with the identity layer such that the embeddings outputted have 512 dimensions (this is arbitary as it just so happens that I used resnet18 for this lightweight task)
# Need to create a transforms operation to pass in raw data into the embedder model
embedderTransforms = transforms.Compose(transforms=[transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

## The custom dataset class used by the resnet18 model to generate the embeddings
class customDataset(Dataset):
    def __init__(self, imgDir, transforms = None):
        self.imgDir = imgDir
        self.transforms = transforms
        '''
        need to collect the image names in order of the frame number, this is importnat because when generating the embeddings from the test frames we 
        want to be able to align the embeddings with the pre-labbeled data (indexes that point to images from the 2 classes)
        '''
        self.imgNames = sorted([f for f in os.listdir(self.imgDir) if f.endswith((".png",".jpg"))], key= lambda f: int(f.split("_")[1].split(".")[0]))
    def __len__(self):
        return len(self.imgNames)
    def __getitem__(self, index):
        imgName = self.imgNames[index]
        fileName = os.path.join(self.imgDir, imgName)
        image = Image.open(fileName).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image

## The logistic Regression dataset class used to format the both embeddings and class's their associated with
class logisticDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
## The logistic regression model 
class LogReg(nn.Module):
    def __init__(self):
        super(LogReg, self).__init__()
        self.layer1 = nn.Linear(in_features=512, out_features=1024)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=1024, out_features=2048)
        self.layer3 = nn.Linear(in_features=2048, out_features=512)
        self.layer4 = nn.Linear(in_features=512, out_features=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        return x


