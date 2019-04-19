from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

def image_transform():
    return transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop((256,256)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])


def find_mean_std(loader):
    mean = 0.
    std = 0.

    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return {'mean': mean, 'std': std}

def un_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    takes a tensor, 
    undo normalization given mean and std,
    transform to numpy,
    transpose dimension from CHW to HWC
    """
    mean = torch.FloatTensor(mean).view(1,3,1,1)
    std = torch.FloatTensor(std).view(1,3,1,1)
    
    image = tensor.cpu().detach()
    image = image*std+mean
    image = image.numpy()
    
    image = np.transpose(image, (0,2,3,1))
    
    #print(np.max(image))
    #print(np.min(image))
    return image

def validate(image, stye_transfer_net, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = transforms.Compose([transforms.Resize(512),transforms.ToTensor(),transforms.Normalize(mean, std)])
    
    image_tensor = transform(image)
    styled_image = stye_transfer_net(image_tensor.unsqueeze(0))
    styled_image = un_normalize(styled_image)
    
    plt.imshow(styled_image[0])
    plt.show()
    
    
    
    