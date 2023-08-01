# pytorch-ppliteSeg

## data

Here is the structure of the "data" directory. The images in the "images" folder are in RGB format, while the labels in the "labels" folder are in grayscale format.

```
data
├── train.txt
├── val.txt
├── images
└── labels
```
## train
use train.sh to train this model,the traine model will be saved 
```
bash train.sh
```

## test
You can utilize the demo.py script to test the model. Please modify the model_path and img_path variables in the demo.py  accordingly.

```
cd tools
python demo.py
```
