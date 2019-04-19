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

def train(net, epochs, batch_size,content_dataset, style_dataset, optimizer, c, s, device, pkl_name, show_every=500, show_images=False):
    content_loader = data.DataLoader(dataset=content_dataset, batch_size=batch_size)
    style_loader = data.DataLoader(dataset=style_dataset, batch_size=1)

    net = net.to(device)
    net.train()

    #criteria = FeatureStyleLoss(c, s).to(device)
    content_loss = ContentLossNet().to(device)
    style_loss = StyleLossNet().to(device)
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
            loss = c_loss + s_loss
            
            #loss = criteria(images, style_images, generated_images)

            loss.backward()
            optimizer.step()
            if b % show_every == 0:
                print('Epoch:{} Batch:{} Loss={:.5f}'.format(e, i+1, loss.item()))
                print('Content_L:{} Style_L:{}'.format(c_loss.item(), s_loss.item()))
                if show_images:
                    show_generated_images(dataset=content_dataset, net=net, device=device)

        torch.save({'state_dict':net.state_dict(),
                    'epoch':e,
                    'c':c,
                    's':s}, (pkl_name+str(e)+'.pth'))
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
    IMAGE_FOLDERS = os.path.join(ROOT,'ImageData')
    SHOW_EVERY = 500
    ANNOTATION_FOLDER = os.path.join(IMAGE_FOLDERS, 'annotations', 'captions_train2014.json')
    TRAIN_IMAGES_FOLDER = os.path.join(IMAGE_FOLDERS, 'train2014')
    STYLE_IMAGE_FOLDER = os.path.join(ROOT, 'ImageData', 'styleimages')
    content_dataset = MSCOCODataset(ANNOTATION_FOLDER, TRAIN_IMAGES_FOLDER, image_transform())
    style_dataset = StyleImageDataset(STYLE_IMAGE_FOLDER, image_transform())
    style_transfer_net = StyleTransferNet()
    optimizer = Adam(style_transfer_net.parameters(), lr=1e-3)
    C = 100000
    S = 440000
    DEVICE = 'cuda'
    PKL_NAME = 'style_transfer'

        

    train(style_transfer_net, EPOCHS, BATCH_SIZE, content_dataset, style_dataset, optimizer, C, S, DEVICE, PKL_NAME)


