import torch
from torchvision.transforms import transforms
import torchvision.transforms._transforms_video as v_transform
from torchvision.transforms.functional import resize


class ConvertTHWCtoTCHW(object):
    """Convert tensor from (T, H, W, C) to (T, C, H, W)
    """

    def __init__(self):
        pass

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 3, 1, 2)

    def __repr__(self):
        return self.__class__.__name__


class ConvertTCHWtoCTHW(object):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)
    """

    def __init__(self):
        pass

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 3, 1, 2)

    def __repr__(self):
        return self.__class__.__name__


class NormalizeImage(object):
    """
    NormalizeImage from [0, 255] to [0, 1]
    """

    def __init__(self):
        pass

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        return vid / 255.0


class VideoTransform:
    def __init__(self, resize_size=(112, 112)):
        self.transforms = transforms.Compose([
            v_transform.ToTensorVideo(),
            transforms.Resize(resize_size)
            # NormalizeImage()
            # ConvertTHWCtoTCHW(),
            # v_transform.NormalizeVideo(mean=mean, std=std),
            # v_transform.CenterCropVideo(crop_size),
        ])

    def __call__(self, x):
        return self.transforms(x)