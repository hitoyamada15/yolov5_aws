import os
import sys
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_suffix, increment_path, non_max_suppression, scale_coords, set_logging
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


def run(source,detect_image):
	with Image.open(source) as save_img:
		imgsz=640 # inference size (pixels)

		classes=None  # filter by class: --class 0, or --class 0 2 3
		agnostic_nms=False  # class-agnostic NMS
		augment=False  # augmented inference
		visualize=False  # visualize features

		hide_labels=False  # hide labels
		hide_conf=False  # hide confidences
		half=False  # use FP16 half-precision inference
		nosave=False  # do not save images/videos
		
		weights='yolov5s.pt' # model.pt path(s)

		iou_thres=0.45  # NMS IOU threshold
		max_det=1000    # maximum detections per image
		conf_thres=0.25  # confidence threshold
		device='cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu

		# save_img = not nosave and not source.endswith('.txt')  # save inference images

		# Initialize
		set_logging()
		device = select_device(device)
		half &= device.type != 'cpu'  # half precision only supported on CUDA

		# Load model
		w = str(weights[0] if isinstance(weights, list) else weights)
		classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
		check_suffix(w, suffixes)  # check weights have acceptable suffix
		pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
		stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
		if pt:
			model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
			stride = int(model.stride.max())  # model stride
			names = model.module.names if hasattr(model, 'module') else model.names  # get class names

		dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
		bs = 1  # batch_size
		vid_path, vid_writer = [None] * bs, [None] * bs

		# Run inference
		if pt and device.type != 'cpu':
			model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
		dt, seen = [0.0, 0.0, 0.0], 0
		for path, img, im0s, vid_cap in dataset:    
			img = torch.from_numpy(img).to(device)
			img = img.half() if half else img.float()  # uint8 to fp16/32
			img /= 255.0  # 0 - 255 to 0.0 - 1.0
			if len(img.shape) == 3:
				img = img[None]  # expand for batch dim

			visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
			pred = model(img, augment=augment, visualize=visualize)[0]

			# NMS
			pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


			# Process predictions
			for i, det in enumerate(pred):  # per image
				seen += 1

				p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

				s += '%gx%g ' % img.shape[2:]  # print string
				gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
				# imc = im0.copy() if save_crop else im0  # for save_crop
				annotator = Annotator(im0, line_width=3, example=str(names))

				if len(det):
					# Rescale boxes from img_size to im0 size
					det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

					# Print results
					for c in det[:, -1].unique():
						n = (det[:, -1] == c).sum()  # detections per class
						s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

					# Write results
					for *xyxy, conf, cls in reversed(det):
						if save_img or save_crop or view_img:  # Add bbox to image
							c = int(cls)  # integer class
							label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
							annotator.box_label(xyxy, label, color=colors(c, True))

		        
				# Stream results
				im0 = annotator.result()

				# Save results (image with detections)
				cv2.imwrite(detect_image, im0)

run("./sample.JPG","./sample_detect_65.jpg")

