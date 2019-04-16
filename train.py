from torch.utils import data
from models import *
from CocoDataset import MSCOCODataset
from styledataset import StyleImageDataset
from utils import image_transform
from torch.optim import Adam
import os


def train(net, epochs, batch_size,content_dataset, style_dataset, optimizer, c, s, device, pkl_name):
    content_loader = data.DataLoader(dataset=content_dataset, batch_size=batch_size)
    style_loader = data.DataLoader(dataset=style_dataset, batch_size=len(style_dataset))

    net = net.to(device)
    style_loss_net = StyleLossNet().to(device)
    content_loss_net = ContentLossNet().to(device)

    print('Training Starts! .......')
    for e in range(epochs):
        for i, images in enumerate(content_loader):
            net.zero_grad()

            images = images.to(device)
            generated_images = net(images)
            style_images = next(iter(style_loader)).to(device)

            loss = c*content_loss_net.content_loss(images, generated_images) \
                   + s*style_loss_net.style_loss(style_images, generated_images)

            loss.backward()

            optimizer.step()

            print('Epoch:{} Batch:{} Loss={:.5f}'.format(e, i, loss.item()))

        torch.save({'state_dict':net.state_dict(),
                    'epoch':e,
                    'c':c,
                    's':s}, (pkl_name+str(e)+'.pth'))

if __name__ == '__main__':
    print(os.getcwd())
    EPOCHS = 1
    BATCH_SIZE = 16
    ANNOTATION_FOLDER = os.path.join('annotations', 'captions_train2014.json')
    TRAIN_IMAGES_FOLDER = 'train2014'
    STYLE_IMAGE_FOLDER = 'styleimages'
    content_dataset = MSCOCODataset(ANNOTATION_FOLDER, TRAIN_IMAGES_FOLDER, image_transform())
    style_dataset = StyleImageDataset(STYLE_IMAGE_FOLDER, image_transform())
    style_transfer_net = StyleTransferNet()
    optimizer = Adam(style_transfer_net.parameters())
    C = 0.7
    S = 0.3
    DEVICE = 'cpu'
    PKL_NAME = 'style_transfer'

    train(style_transfer_net, EPOCHS, BATCH_SIZE, content_dataset, style_dataset, optimizer, C, S, DEVICE, PKL_NAME)


