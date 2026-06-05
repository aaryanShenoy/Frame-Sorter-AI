# [Frame-Sorter-AI]

<!-- 
Add logo image for project here later
-->

<img width="1000" height="250" alt="FRAME-SORTER-AI" src="https://github.com/user-attachments/assets/e1261b8b-3cfc-4b40-be45-56e99eaeeb84" />


## Semi-Supervised Footage Separator Tool (Binary)
A lightweight, computer vision tool that allows you to do 2 cool things:

1. Train a classifier model to separate out footage to your exact (binary) specifications into 2 classes using relatively few sample frames
2. Run a highly accurate inference that automatically sorts frames into designated folders for easy video editing/analysis.

Extra highlights of the tool:

- Uses semi-supervised pipeline, utilising image embeddings from resnet18 to output high quality image embeddings to feed a custom made logistic regression classifier
- Only requires manually sorting around 100 sample frames (for a video with approx 10k frames), the model can then handle the remaining thousands with very high accuracy
- (Note) This tool only works well if you have 2 distinct angles/views/parts to your footage that you are separating


## How The Project Works In A Nutshell

1. The user selects a video to train on
2. Sample frames are extracted from the video and manually separated by the user into `class0` and `class1` folders
3. Training phase reads the class folders, generates embeddings from the frames (via resnet18) and trains a logistic regressor on the embeddings and the labels
4. The resulting trained logistic regressor model is then used for inference

### File Structure Overview 
- `built_models.py` - Contains the Pytorch Neural Network and custom dataset classes
- `training_data_setup.py` - Acts as a helper script to extract the initial sample frames from training video (for user to manually sort)
- `train.py` - Trains the classifier on the manually sorted sample frames
- `inference.py` - Processes the entire video (of the user's choice) and separates the output into 2 folders


## Installation and Setup
**Step 1: Clone/Download the Repository**   

**Step 2: Install Dependencies in the requirements**
Make sure you have python installed and then run the `requirements.txt` file 
```sh
pip install -r requirements.txt
```

## How To Use
**Note**
The files do contain a number of hyperparameters that can be explored (e.g. the number of sample frames extracted/ the logistic neural network config/ the resnet model used for embeddings) but I'll leave this to you as the user to explore within the files if you are interested.

**Step 1: Extract Sample Frames**
Input the file path to your training video into the `training_data_setup.py` script and run the script to get a small collection of sample frames (can be adjusted via hyperparameters, see code)
```sh
python training_data_setup.py
```

**Step 2: Manually Label the Extracted Samples**
The `training_data_setup.py` script also creates a `sampleFramesFolder` with two important subfolders `class0` and `class1`:
- Look at the extracted frames in `sampleFramesFolder`
- Drag and drop ~25-50 frames into `class0`
- Drag and drop ~25-50 frames into `class1`
- (Note) You do NOT need to use all the extracted sample frames in the `sampleFramesFolder` instead just choose 25-50 high quality examples 
- (Note) The 25-50 is just a recommendation, please adjust based on complexity of your footage and use case

**Step 3: Train the Model**
After sorting out a few images train the classifier by running the `train.py` script
```sh
python train.py
```
This will output the trained model's accuracy, precision and recall and save a `trained_classifier.pth` file to your working directory, if you aren't happy with the results try adding more images to your `class0` and `class1` folders from step 2.

**Step 4: Run Inference on the Full Video**
Run the `inference.py` script and the model will separate out the frames in your footage (Note you must provide a target filepath to the .mp4 video you want to extract footage from)
```sh
python inference.py
```
## License
This project is under the MIT License, please see the License file for more details


## Acknowledgements

Big thanks to the following projects:
- resnet18 - Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (Microsoft Research 2015)
- Pytorch - Gregory Chanan, Soumith Chintala, Sam Gross, Adam Paszke (Meta AI 2016)
- OpenCv -  Willow Garage, Itseez (Intel 1999)
- Pillow - Jeffrey A. Clark (2010), Fredrik Lundh (PIL) (1995)

## TODO
- Possibly build a ui for better ease of use for user

