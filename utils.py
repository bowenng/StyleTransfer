from torchvision import transforms


def image_transform():
    return transforms.Compose([transforms.Resize(256),
                               transforms.RandomCrop((256,256)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.4711, 0.4475, 0.4080],[0.2341, 0.2291, 0.2325])])


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