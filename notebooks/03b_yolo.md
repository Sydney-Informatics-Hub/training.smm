# Image Segmentation with YOLO

The [YOLO computer vision model](https://docs.ultralytics.com/) provides the tools and architecture for image segmentation tasks.

It is a command line/python tool, allows substantial flexibility, is open-source, and works on any platform.

Here documented we use YoloV8 in its simplest form. Additional tools and advanced usage including YoloV5 can be found on the [main repo](https://github.com/Sydney-Informatics-Hub/Microscopy-Segmentation/tree/main)

## Setup your Python environment

Start an Anaconda prompt or similar and execute the following commands
```
conda create -n yolo python=3.1
conda activate yolo
pip install ultralytics==8.0.227
pip install scikit-learn==1.3.2
```
This will install all required packages.

## Labels to YOLO format

YOLO models need the data to be in a specific format and folder structure.
Use the following tool to do this in one step assuming you created labels in [AnyLabeling]. If you created labels using some other method, check the [tools] folder for different conversion scripts.

```
python tools/anylableing2yolo.py --class_labels '{"mito": 0, "float": 1}' --input_dir 'input_data' --output_dir 'datasets/mito_data' --split_ratio 0.2
```

This will create a new folder in `datasets` with the following structure:
```
mito_data/
├── images
│   ├── train
│   │   ├── image01.tif
│   │   ├── image02.tif
│   │   ├── image03.tif
...
│   ├── validate
│   │   ├── image17.tif
│   │   ├── image18.tif
│   │   ├── image19.tif
...
├── labels
│   ├── train
│   │   ├── image01.txt
│   │   ├── image02.txt
│   │   ├── image03.txt
...
│   ├── validate
│   │   ├── image17.tif
│   │   ├── image18.tif
│   │   ├── image19.tif
```

## Training your model

Now you can execute a task that will use a pre-trained model as a starting point and use your data to refine the model specific to your prediction tasks.

On your command line execute the following

```
yolo detect train data=mito.yaml model=yolov8m-seg.pt
```

See all the available command line options here https://docs.ultralytics.com/modes/train/#arguments

The choice of `yolov8m-seg.pt` is based on the performance https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes and being specifically trained for semantic segmentation. Feel free to experiment with other models more tuned for your problem.

The structure of `mito.yaml` file is as follows:

```
path: ../datasets/mito_data
train: images/train  
val: images/validate
test:  # test images (optional)

names:
  0: mito
  1: fat
```

Additional classes should be added (removed) as needed, and the paths should point to where your data are.

If all goes well this will produce the weights of a trained model in the default output folder (`runs/segment/train/weights/best.pt`).

We will now use this file to make predictions on new data!

## Inference

The final step is to make predictions on new data

```
yolo predict model=runs/segment/train/weights/best.pt source=path/to/raw_images/
```
This will output data. However, we need to customise the output using Python to get a clean image mask similar to Avizo output, and probably what most users want for microscopy. The prediction script is probably what we want.

```
python predict.py model=runs/segment/train/weights/best.pt source=path/to/raw_images/
```

Where predict.py contains
```
from ultralytics import YOLO

# Load a pretrained YOLOv8n-seg Segment model
model = YOLO('runs/segment/train/weights/best.pt')

# Run inference on an image
results = model(=path/to/raw_images/)  # results list

# View results
for r in results:
    print(r.masks)  # print the Masks object containing the detected instance masks
```
