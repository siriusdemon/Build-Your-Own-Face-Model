import torch
from torchvision import transforms as T


class Config:
    # preprocess
    insize = [416, 416]
    channels = 3
    downscale = 4
    sigma = 2.65

    train_transforms = T.Compose([
        T.ColorJitter(0.5, 0.5, 0.5, 0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
    ])

    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
    ])

    # dataset
    dataroot = '/data/WIDER_train/images'
    annfile = '/data/retinaface_gt_v1.1/train/label.txt'

    # checkpoints
    checkpoints = 'checkpoints'
    restore = False
    restore_model = 'final.pth'

    # training
    epoch = 90
    lr = 5e-4
    batch_size = 24
    pin_memory = True
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # inference
    threshold = 0.5
