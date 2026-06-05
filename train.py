import numpy as np
import torch.nn as nn
import torch
import torch.optim as optimizer
from torch.utils.data.dataloader import DataLoader
from built_models import customDataset, embedderTransforms, embedderModel, logisticDataset, LogReg
from training_data_setup import class0Dir, class1Dir


# creating the datasets and the loaders (can change batch_size parameter if needed)
class0ImgSet = customDataset(imgDir=class0Dir, transforms=embedderTransforms)
class1ImgSet = customDataset(imgDir=class1Dir, transforms=embedderTransforms)

class0ImgLoader = DataLoader(dataset=class0ImgSet, batch_size=64)
class1ImgLoader = DataLoader(dataset=class1ImgSet, batch_size=64)

## getting the embeddinhg and the labels for each class via the embedder model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedderModel.to(device)

embeddingsClass0List = []
embeddingsClass1List = []

embedderModel.eval()
with torch.no_grad():
    for frameClass0, frameClass1 in zip(class0ImgLoader, class1ImgLoader):
        frameClass0 = frameClass0.to(device)
        frameClass1 = frameClass1.to(device)

        embeddingFrameClass0 = embedderModel(frameClass0)
        embeddingFrameClass1 = embedderModel(frameClass1)

        embeddingsClass0List.append(embeddingFrameClass0.cpu().numpy())
        embeddingsClass1List.append(embeddingFrameClass1.cpu().numpy())
    
embeddingsClass0List = np.concatenate(embeddingsClass0List, axis = 0)
embeddingsClass1List = np.concatenate(embeddingsClass1List, axis = 0)


labelsClass0List = np.full(len(embeddingsClass0List), 0)
labelsClass1List = np.full(len(embeddingsClass1List), 1)

## combining the embeddings and labels into a unified list for training/testing and shuffling 
embeddingsList = np.concatenate([embeddingsClass0List, embeddingsClass1List], axis=0)
labelsList = np.concatenate([labelsClass0List,labelsClass1List], axis=0)

# the aligned shuffling process
rng = np.random.default_rng()
permutations = rng.permutation(len(embeddingsList))

embeddingsListShuffled = embeddingsList[permutations]
labelsListShuffled = labelsList[permutations]

# train/test split to check results
trainRatio = 0.6 
splitIndex = int(trainRatio * len(embeddingsList))

trainSetLabelledX, trainSetLabelledY = embeddingsListShuffled[:splitIndex], labelsListShuffled[:splitIndex]
testSetLabelledX, testSetLabelledY = embeddingsListShuffled[splitIndex:], labelsListShuffled[splitIndex:]

# creating the training/testing datasets and the dataLoaders
trainSet = logisticDataset(data=trainSetLabelledX, labels=trainSetLabelledY)
testSet = logisticDataset(data=testSetLabelledX, labels=testSetLabelledY)

trainLoader = DataLoader(dataset=trainSet,batch_size=64)
testLoader = DataLoader(dataset=testSet,batch_size=1) # batch size set to 1 for straightforward test


### Now for the training part + results

model = LogReg()
criterion = nn.BCEWithLogitsLoss()
optimiser = optimizer.Adam(model.parameters(), lr = 0.01)
epochs = 100

model.to(device)

model.train()
for _ in range(epochs):
    for batchX, batchY in trainLoader:
        batchX = batchX.to(device)
        batchY = batchY.to(device)

        pred_batchY = model(batchX).squeeze(1)
        loss = criterion(pred_batchY, batchY)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

# gathering the test data
truePositives = 0
falsePositives = 0

trueNegatives = 0
falseNegatives = 0

# evaluting the model
model.eval()
with torch.no_grad():
    for Xtest, Ytest in testLoader:
        Xtest = Xtest.to(device)
        Ytest = Ytest.to(device)

        predYtest = model(Xtest)
        prob = torch.sigmoid(predYtest)
        predYclass = (prob > 0.5).float()
        Ytest = Ytest.item()
        if (predYclass == 1):
            if (Ytest == 1):
                truePositives += 1
            else: #i.e. Ytest == 0 meaning model did a false positive
                falsePositives += 1
        else: # predYclass == 0
            if (Ytest == 0):
                trueNegatives += 1
            else: # i.e. Ytest == 1 and model predicted 0 (false negative)
                falseNegatives += 1

total = truePositives + falsePositives + trueNegatives + falseNegatives


# getting the main stats
print(f"Accuracy = {100 * (truePositives + trueNegatives)/total}")
print(f"Precision = {100 * (truePositives/(truePositives + falsePositives))}")
print(f"Recall = {100 * (truePositives/(truePositives + falseNegatives))}") 

# saving the model 
torch.save(model.state_dict(), "trained_classifier.pth")





