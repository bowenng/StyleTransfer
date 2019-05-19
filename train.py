from torch.utils import data
from models import *
from CocoDataset import MSCOCODataset
from styledataset import StyleImageDataset
from utils import image_transform
from utils import un_normalize
from torch.optim import Adam
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

def train(net, epochs, batch_size,content_dataset, style_dataset, optimizer, c, s, color, device, pkl_name, show_every=500):
    content_loader = data.DataLoader(dataset=content_dataset, batch_size=batch_size)
    style_loader = data.DataLoader(dataset=style_dataset, batch_size=1)

    net = net.to(device)
    net.train()

    #criteria = FeatureStyleLoss(c, s).to(device)
    content_loss = ContentLossNet().to(device)
    style_loss = StyleLossNet().to(device)
    color_loss = ColorSimilarityLoss().to(device)
    b = 0
    print('Training Starts! .......')
    for e in range(1, epochs+1):
        for i, images in enumerate(content_loader):
            b += 1
            optimizer.zero_grad()

            images = images.to(device)
            generated_images = net(images).to(device)
            style_images = next(iter(style_loader)).to(device)
            
            c_loss = c*content_loss(images, generated_images)
            s_loss = s*style_loss(style_images, generated_images)
            clr_loss = color*color_loss(images, generated_images)
            
            loss = c_loss + s_loss + clr_loss
            
            

            loss.backward()
            optimizer.step()
            if b % show_every == 0:
                print('Epoch:{} Batch:{} Loss={:.5f}'.format(e, i+1, loss.item()))
                print('Content_L:{} Style_L:{}'.format(c_loss.item(), s_loss.item()))
                print('Color_L:{}'.format(clr_loss.item()))

        torch.save({'state_dict':net.state_dict(),
                    'epoch':e,
                    'c':c,
                    's':s,
                    'color': color,
                   'content_loss': c_loss.item(),
                   'style_loss': s_loss.item(),
                   'color_loss': clr_loss.item()}, (pkl_name+str(e)+'.pth'))
    print('Training Finished!')




def show_generated_images(dataset, net, device,show_n=5):
    image_idx = np.random.choice(len(dataset), show_n)
    image_idx
    images = []
    for idx in image_idx:
        images.append(dataset[idx])

    images = torch.stack(images).to(device)
    original_images = un_normalize(images)
    generated_images = un_normalize(net(images))

    
    fig, axes = plt.subplots(2, len(original_images))
    
    for i in range(len(original_images)):
        axes[0, i].imshow(original_images[i])
        axes[1, i].imshow(generated_images[i])
        
    plt.show()

if __name__ == '__main__':

    EPOCHS = 2
    BATCH_SIZE = 4
    ROOT = os.getcwd()
    SHOW_EVERY = 500
    TRAIN_IMAGES_FOLDER = os.path.join(ROOT, 'train2014')
    STYLE_IMAGE_FOLDER = os.path.join(ROOT, 'StyleImages')
    content_dataset = MSCOCODataset(TRAIN_IMAGES_FOLDER, image_transform())
    style_dataset = StyleImageDataset(STYLE_IMAGE_FOLDER, image_transform())
    style_transfer_net = StyleTransferNet()
    optimizer = Adam(style_transfer_net.parameters(), lr=1e-3)
    C = 100000.0
    S = 700000.0
    COLOR = 20000.0
    DEVICE = 'cuda'
    PKL_NAME = 'vangogh_YIQ'
        

    train(style_transfer_net, EPOCHS, BATCH_SIZE, content_dataset, style_dataset, optimizer, C, S, COLOR, DEVICE, PKL_NAME)


