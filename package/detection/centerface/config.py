import torch
from torchvision import transforms as T


class Config:
    # preprocess
    insize = [416, 416]
    channels = 3
    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
    ])

    restore_model = 'final.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # inference
    threshold = 0.5