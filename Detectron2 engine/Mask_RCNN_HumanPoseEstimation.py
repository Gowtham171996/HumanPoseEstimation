import detectron2
import ctypes
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    predictor = DefaultPredictor(cfg)
    outputs = predictor(frame)
    v = Visualizer(frame[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Display the resulting frame
    cv2.imshow('frame',v.get_image()[:, :, ::-1])
   
    # Write the output frame to disk
    #writer.write(v.get_image()[:, :, ::-1])
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
ctypes.windll.user32.MessageBoxW(0, "The video has been succefully completed", "Finished", 1)

'''
im = cv2.imread("girl.jpeg")
cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image",600,600)
cv2.imshow("Image",im)

cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)


v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.namedWindow("Image1",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image1",600,600)
cv2.imshow("Image1",v.get_image()[:, :, ::-1])
'''
