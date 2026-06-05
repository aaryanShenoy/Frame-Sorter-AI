import cv2
import os

'''
Run the sampleFrameGenerator below (with your desired inputs) to extract the sample frames into a folder, HOWEVER, you will then need to create 2 separate folders
such that each folder contains the distinct angle/view/part of the footage you want the final model to be able to learn to extract from, 
you don't need too many examples per folder, but 25-50 images in each folder is a good benchmark (try make it a 50/50 split between the classes)

Example file structure created is:
sampleFrameFolder/
    |__ class0/
    |
    |__ class1/

You just need to transfer the frames from the sampleFramesFolder into the sub folders manually as part of the labelling process
'''

# generates the sample frames 
def sampleFrameGenerator(folderName = "sampleFramesFolder", fileName = None, sampleRate = 30):
    if (not os.path.exists(folderName)) and fileName:
        os.mkdir(path=folderName)
        # running through each frame via cv2
        cap = cv2.VideoCapture(filename=fileName)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if (i % sampleRate == 0): 
                saveFileName = os.path.join(folderName,f"frame_{i}.jpg")
                cv2.imwrite(filename=saveFileName,img=frame)
            i += 1


        cap.release()

    else:
        print("Invalid Input")


folderName = "sampleFramesFolder"

''' Include your filePath here that points to the data you want the classifier model to be trained on '''
fileName = None

sampleRate = 30
sampleFrameGenerator(folderName=folderName, fileName=fileName, sampleRate=sampleRate)


class0Dir = os.path.join(folderName,"class0")
class1Dir = os.path.join(folderName, "class1")

if not os.path.exists(class0Dir):
    os.mkdir(class0Dir)
if not os.path.exists(class1Dir):
    os.mkdir(class1Dir)
    
