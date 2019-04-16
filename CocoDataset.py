import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
import os
import numpy as np



class MSCOCODataset(data.Dataset):

    def __init__(self, annotation_file: str, image_folder: str, transform=None) -> None:
        super().__init__()

        self.annotation_file = annotation_file
        self.image_folder = image_folder
        self.transform = transform

        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_file = self.coco.loadImgs([image_id])[0]['file_name']
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_ids)
