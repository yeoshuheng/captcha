import cv2
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torch
from torch.utils.data import Dataset

from commons import CHARSET, CHAR_TO_IDX

class CaptchaPreprocess:
    def __init__(self, img_height=64, img_width=200, augment=False):
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        self.normalize = T.Normalize(mean=[0.5], std=[0.5])

    def adaptive_threshold(self, img):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def __call__(self, img):
        img = np.array(img.convert('L'))    
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # denoise
        img = cv2.fastNlMeansDenoising(img, None, h=10)
        
        # apply clahe to increase contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # adaptive thresholding
        if np.random.rand() < 0.3 and self.augment:
            thresh = self.adaptive_threshold(img)
            img = cv2.addWeighted(img, 0.7, thresh, 0.3, 0)
        
        if self.augment:
            # rotation
            if np.random.rand() < 0.4:
                angle = np.random.uniform(-3, 3)
                M = cv2.getRotationMatrix2D((self.img_width/2, self.img_height/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (self.img_width, self.img_height), 
                                    borderMode=cv2.BORDER_REPLICATE)
            
            # brightness / contrast
            if np.random.rand() < 0.5:
                alpha = np.random.uniform(0.8, 1.2)  # contrast
                beta = np.random.uniform(-10, 10)    # brightness
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            
            # noise
            if np.random.rand() < 0.3:
                noise = np.random.normal(0, 3, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # blur
            if np.random.rand() < 0.2:
                kernel_size = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
            # sharpening
            if np.random.rand() < 0.3:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                img = cv2.filter2D(img, -1, kernel)
                img = np.clip(img, 0, 255).astype(np.uint8)
        
        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        img = self.normalize(img)
        return img

class CaptchaDataset(Dataset):
    def __init__(self, image_dir, img_height=64, img_width=200, max_label_len=10, augment=False):
        self.image_paths = []
        self.labels = []
        self.max_label_len = max_label_len
        
        skipped = 0
        for path in Path(image_dir).glob("*.png"):
            label = path.stem.split('-')[0]
            
            if len(label) > max_label_len or len(label) == 0:
                skipped += 1
                continue
            if not all(c in CHARSET for c in label):
                skipped += 1
                continue
                
            self.image_paths.append(str(path))
            self.labels.append(label)
        
        if skipped > 0:
            print(f"Skipped {skipped} invalid samples")
        
        self.transform = CaptchaPreprocess(img_height, img_width, augment=augment)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = self.transform(img)
        
        label = self.labels[idx]
        label_indices = torch.tensor([CHAR_TO_IDX[c] for c in label], dtype=torch.long)
        
        return img, label_indices, label

def collate_fn(batch):
    images, labels, label_strs = zip(*batch)
    images = torch.stack(images, dim=0)
    labels_concat = torch.cat(labels)
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return images, labels_concat, label_lengths, label_strs