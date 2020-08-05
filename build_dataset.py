import os
from utils import config
from imutils import paths
import shutil

for split in (config.TRAIN,config.TEST,config.VAL):
    
    #Grabbing the image paths listed in the required directory
    path=os.path.sep.join([config.ORIG_INPUT_DATASET,split])
    imagePaths=list(paths.list_images(path))
    
    for p in imagePaths:
        #extracting the class label
        filename = p.split(os.path.sep)[-1]
        label = config.CLASSES[int(filename.split("_")[0])]
        
        #construct the path to the output directory
        dirPath = os.path.sep.join([config.BASE_PATH, split, label])
        
        #If directory not exists,make a new one
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
            
        #construct the path to the output image file and copy it    
        imagePath = os.path.sep.join([dirPath, filename])
        shutil.copy2(p, imagePath)
    