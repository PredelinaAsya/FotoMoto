# FotoMoto

## Intro

The goal of this project is to replace the manual sorting of photos from the motorcycle race by individual pilots with an automatic one in order to speed up and facilitate this stage of the photographers' work.

## How to run the code?

1. Create new venv and install required packages: ```pip install -r requirements.txt```

2. Basic run with default hparams: ```python3 main.py --input_folder {Path to input folder with photos}```

    As a result, `results` directory will be created with subdirectories with photos sorted by individual pilots.

3. You can also find examples of running individual stages of the algorithm in saved notebooks (`notebooks` folder)

## Description of the solution

The proposed solution consists of the following steps:

1. get masks from the segmentation model (`yolov8n-seg`) with post-processing in order to correctly combine the masks of pilots and motorcycles (matching algorithm)

2. build vector representations of motorcycle pilots based on the pixel values of masks (2 different approaches have been implemented through the construction of color histograms)

3. separation by individual pilots using clustering on constructed vector representations (the K-Means method is used)

## Demo

...
