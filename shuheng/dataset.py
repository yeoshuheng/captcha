import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torchvision.io import read_image
import string
from typing_extensions import List, Dict, Tuple
from pathlib import Path

CHARSET = string.ascii_lowercase + string.ascii_uppercase + string.digits

def get_image_paths_and_labels(relative_dir: str) -> Tuple[List[str], List[str]]:
    
    file_paths = []
    labels = []

    for path in Path(relative_dir).iterdir():
        if path.is_file() and path.suffix == '.png':
            file_paths.append(str(path.resolve()))
            
            name = path.stem
            label = name.split('-')[0]
            labels.append(label)

    return file_paths, labels

class CaptchaImageDataset(Dataset):
    _int_to_char : Dict[int, str] = {i + 1: c for i, c in enumerate(CHARSET)}
    _char_to_int : Dict[str, int] =  {c: i + 1 for i, c in enumerate(CHARSET)}
    
    def __init__(self, file_paths: List[str], captcha: List[str], img_height: int=32, img_width: int = 128, normalisation_factor:float=0.5):
        self._file_paths = file_paths
        self._img_height = img_height
        self._img_width = img_width
        self._captcha = captcha

        self.transform = T.Compose(
            [
                T.Grayscale(1),
                T.Resize((self._img_height, self._img_width)),
                T.Normalize((normalisation_factor,), (normalisation_factor, ))
            ]
        )
    
    def __len__(self):
        return len(self._file_paths)

    def __getitem__(self, i):
        img = read_image(self._file_paths[i]).float()

        if img.shape[0] == 4:
            img = img[:3, :, :] 
            
        label =  self._captcha[i]
        
        img = self.transform(img)
        label = torch.Tensor([self._char_to_int[c] for c in label], dtype=torch.long)

        return img, label, len(label)