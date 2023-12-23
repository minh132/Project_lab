import os
import json
import math
import numpy as np
import torch
# from monai import transforms, data
# import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 
# import pandas as pd
import albumentations as A
from pycocotools.coco import COCO
from torchvision import datasets, models
from PIL import Image
from albumentations.pytorch import ToTensorV2
import cv2
import copy
from torchvision.transforms import transforms as T
import sys
sys.path.append('/home/minh/Documents/DiffUnet-main')
from tool.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(600, 600), 
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(600, 600),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform


class AquariumDetection(datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        split = "train",
        transform= None,
        target_transform = None,
        transforms = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.split = split
        self.coco = COCO(os.path.join(root, split, "_annotations.coco.json"))
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        print(os.path.join(self.root, self.split, path))
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def _coco_box_to_bbox(box):
        return np.array(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32
        )

    def __getitem__(self, index: int):
        num_classes=7
        id = self.ids[index]
        image = self._load_image(id)
        target = copy.deepcopy(self._load_target(id))
        boxes = [t['bbox'] + [t['category_id']] for t in target]
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
        num_objs=len(target)
        image = transformed['image']
        boxes = transformed['bboxes']
        new_boxes = []
        # boxes = torch.tensor(new_boxes, dtype=torch.float32)
        _, h, w = image.shape
        draw_gaussian=draw_umich_gaussian 
        heatmap = torch.zeros(
            (num_classes, 600, 600), dtype=torch.float32
        )     
        for k in range(num_objs):
            inf=boxes[k]
            cls_id=int(inf[4])
            print(cls_id)
            h,w=inf[2],inf[3]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = torch.FloatTensor(
                    [(inf[0] + inf[2]) / 2, (inf[1] + inf[3]) / 2]
                )
                ct_int = ct.to(torch.int32)

                draw_gaussian(heatmap[cls_id], ct_int, radius)
        print(heatmap.shape)
        targ = {}
        targ["heatmap"] = heatmap
        targ["image"]=image
        print(image)
        # targ["labels"] = torch.tensor([t["category_id"]  for t in target], dtype=torch.int64)
        # targ["image_id"] = torch.tensor([t["image_id"]  for t in target])
        # targ["img_scale"] = torch.tensor([1.0])
        # targ['img_size'] = (h, w)
        
        
        return targ


    def __len__(self) -> int:
        return len(self.ids)



if __name__ == '__main__':
    dataset_path='/home/minh/Documents/DiffUnet-main/data'
    train_dataset = AquariumDetection(root=dataset_path, split="train",transforms=get_transforms(True))
    print(train_dataset[0])
    val_dataset = AquariumDetection(root=dataset_path, split="valid", transforms=get_transforms(False))
    print(len(val_dataset))
    test_dataset = AquariumDetection(root=dataset_path, split="test", transforms=get_transforms(False))