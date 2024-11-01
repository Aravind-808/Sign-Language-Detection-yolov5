# Sign-Language-Detection-yolov5
Detecting sign language and their corresponding alphabets using the yolov5m model.

More about yolov5: [https://blog.roboflow.com/yolov5-improvements-and-evaluation/](url)

There are some steps that need to be followed in order to be able to train and implement your own yolov5 model. Here I will go through what I did. This is very rough and only for basic understanding, so i recommend looking up tutorials online as the steps may vary for different devices.

This is mostly for my own documentation purposes for future reference.

## Getting all the required stuff
### Virtual Environment
Creating a virtual environment, though optional is advantageous as you can download project specific versions of packages without clashing with the versions in the global environment.
So, create one in your project folder before proceeding :)

### Cloning the ultralytics repo
Ultralytics has a yolov5 repository that you'd have to clone before proceeding. Use the command below
```
git clone https://github.com/ultralytics/yolov5
cd yolov5
```
Once the repo has been cloned, I installed the requirements using this command
```
pip install -U -r yolov5/requirements.txt
```
Now comes the part where we download and use the dataset.

### GPU utilization
Before beginning with that, there is one more requirement. If you have an NVIDIA GPU, install CUDA as it allows the training to use the GPU rather than the CPU, which will take more time.

First, check your NVIDIA CUDA version using the command ```nvidia-smi```

[https://developer.nvidia.com/cuda-downloads](url) 
Download the CUDA toolkit and set it up here. Make sure to refer online tutorials for a smooth setup, as i cant cover it all here. Check CUDA with ```nvcc --version```

Also use the command ```pip show torch``` to check if your pytorch supports gpu. If not, install a pytorch with the latest cuda version from its official website.

After following these steps, I was able to use my GPU to train the model. This step may vary for different systems so please look up system specific tutorials.

## Dataset
### Downloading the dataset
I used roboflow's American Sign Language dataset from [https://public.roboflow.com/object-detection](url). If youre new to this, i recommend doing the same, but with a different dataset.
We will be using the dataset in the ```YOLOv5 Pytorch``` format, which will contain annotations for each image (.txt) and a data.yaml file which is important for training.

Click on the dataset > raw > show download code, and it should return something like this

```
curl -L "https://public.roboflow.com/ds/<Your Unique Private Key>" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```
If that doesnt work, try this
```
Invoke-WebRequest -Uri "https://public.roboflow.com/ds/<Your Unique Private Key>" -OutFile "roboflow.zip"
Expand-Archive -Path "roboflow.zip" -DestinationPath "."
Remove-Item "roboflow.zip"
```
Now the train and test filders will have some images, annotations and I got a data.yaml file, that contains data that looked something like this:
```
train: ./data/dataset_folder/images/train  
val: ./data/dataset_folder/images/val
nc: <number_of_classes>
names: ["class_1", "class_2", ...]              
```
This file contains the path of the training and testing images, along with the number of classes and their corresponding class names.

## Training
### epochs
Epoch is the number of times a training dataset passes through an algorithm. The number of epochs may vary depending on the size of the dataset, performance of machine, overfitting, etc.

### batch
An epoch is made up of batches, which are small groups of data that are used to feed the model.

### img
The size of the image, typically 640.

### weights
If youre gonna use a nano (yolov5n.pt), small (yolov5s.pt), medium (yolov5m.pt) or large (yolov5l.pt) yolo model. I used the medium one.

### device
Not specifying this will use the CPU, which is very slow. If there is a recognized GPU, you can set it to 1.

### data
The data.yaml file.

The final command should look something like this
```
python train.py --img 640 --batch 16 --epochs 100 --data ./data.yaml --weights yolov5m.pt --device 0
```
Finishing training took me about 4 to 5 hours. So go watch a movie or something!

## Testing with a sample image

You can test the trained model with a sample image
```
python detect.py --weights path_to_best.pt --img-size 640 --source path_to_image
```
It will save the result in runs/detect/

I wrote a python script for live detection, too.

And that's it!! This is how you train your own yolov5 model. 
Bye-Bye!!
