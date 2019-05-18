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

def validate(image, style_transfer_net, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cpu'):
    transform = transforms.Compose([transforms.Resize(512),transforms.ToTensor(),transforms.Normalize(mean, std)])
    
    style_transfer_net.to(device)
    image_tensor = transform(image).to(device)
    styled_image = style_transfer_net(image_tensor.unsqueeze(0))
    styled_image = un_normalize(styled_image.to('cpu').detach())
    
    fig, axes = plt.subplots(1,2,figsize=(30,30))
    axes[0].imshow(image)
    axes[1].imshow(styled_image[0])
    plt.show()
    
def rgb_to_ycbcr(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cuda'):
    """
    ref: https://discuss.pytorch.org/t/how-to-change-a-batch-rgb-images-to-ycbcr-images-during-training/3799 by smth
    """
    mean = torch.tensor(mean, device=device).view(1,3,1,1)
    std = torch.tensor(std, device=device).view(1,3,1,1)
    
    images = images*std + mean
    
    KR = 0.299
    KG = 0.587
    KB = 0.114
    
    # input is mini-batch N x 3 x H x W of an RGB image
    output = images.new(*images.size())
    
    # similarly write output[:, 1, :, :] and output[:, 2, :, :] using formulas from https://en.wikipedia.org/wiki/YCbCr
    output[:, 0, :, :] = images[:, 0, :, :] * KR + images[:, 1, :, :] * KG + images[:, 2, :, :] * KB #Y
    #output[:, 1, :, :] = images[:, 0, :, :] * -0.14713 + images[:, 1, :, :] * 0.28886+ images[:, 2, :, :] * 0.436 #Cb
    #output[:, 2, :, :] = images[:, 0, :, :] * 0.615 + images[:, 1, :, :] * -0.51499 + images[:, 2, :, :] * -0.10001 #Cr
    
    # ref: https://www.eembc.org/techlit/datasheets/yiq_consumer.pdf, 
    # https://blogs.mathworks.com/cleve/2016/12/12/my-first-matrix-rgb-yiq-and-color-cubes/
    output[:, 1, :, :] = images[:, 0, :, :] * 0.5959 - images[:, 1, :, :] * 0.2744 - images[:, 2, :, :] * 0.3216 #I
    output[:, 2, :, :] = images[:, 0, :, :] * 0.2115 - images[:, 1, :, :] * -0.5229 + images[:, 2, :, :] * 0.3114 #Q
    
    
    return output