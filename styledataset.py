from torch.utils import data
from PIL import Image
import os


class StyleImageDataset(data.Dataset):

    def __init__(self, image_folder, transform=None) -> None:
        super().__init__()
        files = os.listdir(image_folder)
        self.image_folder = image_folder
        self.image_files = [file for file in files if self.is_image(file)]# store image file names in a list
        self.transform = transform
    
    @staticmethod
    def is_image(file):
        return file.lower().endswith(('.jpg', '.png', '.jpeg'))
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_folder, self.image_files[index])).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_files)

