import os
import numpy as np
import torch
from PIL import Image
import open_clip
from torch.utils.data import Dataset, DataLoader
import pickle
import argparse
from torchvision.transforms import Resize, CenterCrop, Compose
from tqdm import tqdm

from utils import *

class PatternNet(Dataset):
    def __init__(self, input_filename, image_transforms, root=None):
        with open(input_filename, 'r') as f:
            lines = f.readlines()
        filenames = [line.strip() for line in lines]
        self.images = [name.split(" ")[0] for name in filenames] 
        self.labels = [name.split(" ")[2] for name in filenames]
        self.image_transforms = image_transforms
        self.root = root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.root is not None:
            img_path = os.path.join(self.root, str(self.images[idx]))
        else:
            img_path = str(self.images[idx])
        image = self.image_transforms(Image.open(img_path).convert("RGB"))
        label = self.labels[idx]
        return image, label, img_path

def save_dataset(model, dataloader, path_save):
    all_image_features, all_labels, all_image_paths = [], [], []
    with torch.no_grad():
        for images, labels, img_paths in tqdm(dataloader, desc="Processing batches"):
            images = images.cuda(non_blocking=True)
            image_features = model.encode_image(images)           
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
            all_image_features.append(image_features)
            all_labels.extend(labels)
            all_image_paths.extend(img_paths)
        
        all_image_features = torch.cat(all_image_features, dim=0)
        dict_save = {
            'feats': all_image_features.data.cpu().numpy(),
            'labels': all_labels,
            'paths': all_image_paths
        }
        directory = os.path.dirname(path_save)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path_save, "wb") as f:
            print('Writing pickle file...')
            pickle.dump(dict_save, f)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Extracting features from the model and saving them into pickle files.')
    parser.add_argument('--model_name', type=str, default='clip', choices=['remoteclip', 'clip'], help='pre-trained model')
    parser.add_argument('--model_type', type=str, default='ViT-L-14', choices=['RN50', 'ViT-B-32', 'ViT-L-14'], help='pre-trained model type')
    parser.add_argument('--dataset', type=str, default='patternnet', choices=['dlrsd', 'patternnet', 'seasons'], help='choose dataset')
    parser.add_argument('--dataset_path', type=str, default='/mnt/datalv/bill/datasets/data/PatternNet/', help='PatternNet dataset path')
    parser.add_argument('--size', type=int, default=224, help='resize and crop size')
    parser.add_argument('--batch_size', type=int, default=128, help='dataloader batch size')
    args = parser.parse_args()

    # Load model and tokenizer
    model, preprocess_images, tokenizer = load_model(args.model_name, args.model_type)

    # Load dataset, extract and save features
    if args.dataset == 'patternnet':
        full_dataset_path = os.path.join(args.dataset_path, 'patternnet.csv')
        full_dataset = PatternNet(full_dataset_path, image_transforms=preprocess_images, root=args.dataset_path)
        full_dataloader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        
        save_path = os.path.join(args.dataset_path, 'features', f'patternnet_{args.model_name}.pkl')
        save_dataset(model, full_dataloader, save_path)