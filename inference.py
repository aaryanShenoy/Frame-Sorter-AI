import cv2
import os
import torch
from PIL import Image
from built_models import embedderTransforms, embedderModel, LogReg
from training_data_setup import class0Dir, class1Dir


''' Input the video you want to separate (can be the same you you trained on as the training phase trained on a few samples)
 but you can now use the completed classifier model to seprate footage from the entire video hence semi-supervised'''
videoFileRelativePath = None # e.g. video.mp4  etc. assuming its in the same folder as the code


videoFileName = videoFileRelativePath.split('.')[0]

if videoFileRelativePath:
    # generating the folders to separate the frames into
    finalSaveFolderName = f"separatedFrame_{videoFileName}"

    if not os.path.exists(finalSaveFolderName):
        os.mkdir(finalSaveFolderName)
    
    class0FramesFolder = os.path.join(finalSaveFolderName, "class0")
    class1FramesFolder = os.path.join(finalSaveFolderName, "class1")

    if not os.path.exists(class0FramesFolder):
        os.mkdir(class0FramesFolder)
    if not os.path.exists(class1FramesFolder):
        os.mkdir(class1FramesFolder)

    '''Now we run inference and handling:
       -> Run through all the frames via cv2 cap
       -> Run each frame through the embedding model (modified resnet18) accounting for PIL imaging and transforms
       -> Run the embeddinging through the trained LogReg model to output either a 0 or 1 (classes)
       -> Use the model output to sort the frame into the correct folder'''


    # loading trained model (if steps have been completed) and the embedder model
    model = LogReg()
    model.load_state_dict(torch.load("trained_classifier.pth"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    embedderModel.to(device)
    embedderModel.eval()

    # opening and running through each frame via cv2
    cap = cv2.VideoCapture(filename=videoFileRelativePath)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frameName = f"frame_{i}.jpg"

        frameImage = Image.fromarray(frame).convert('RGB')
        frameImage = embedderTransforms(frameImage).unsqueeze(0).to(device)
        embedding = embedderModel(frameImage)
        resultProbability = torch.sigmoid(model(embedding))
        result = (resultProbability > 0.5).float()
        if result.item() == 0:
            saveFileName = os.path.join(class1FramesFolder, frameName)
        else: # i.e result.item() == 0
            saveFileName = os.path.join(class0FramesFolder, frameName)
        
        cv2.imwrite(filename=saveFileName, img=frame)
        i += 1

    cap.release()


else:
    print("Please input a video file to separate footage from")







