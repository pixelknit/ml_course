from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TEST = ("car_dataset", )
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  

cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

im = cv2.imread("/home/fpshell/pictures/inference/cars/car_test.jpg")
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("car_dataset"), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imshow("Car Detection", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
