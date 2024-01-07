import os
import numpy as np
import torch
from PIL import Image
import open_clip
from torch.utils.data import Dataset, DataLoader
import pickle
import argparse
from torchvision.transforms import Resize, CenterCrop, Compose


class ImageSegMapsDataset(Dataset):
    def __init__(self, input_filename, image_transforms, withsegmaps=True, root=None):
        self.withsegmaps = withsegmaps
        with open(input_filename, 'r') as f:
            lines = f.readlines()
        filenames = [line.strip() for line in lines]
        self.images = [name.split(" ")[0] for name in filenames] 
        if self.withsegmaps:
            segmaps_filenames = [x.replace('Images_png', 'Labels') for x in filenames]
            self.segmaps = [name.split(" ")[0] for name in segmaps_filenames]
            self.classes = {0: "airplane", 1: "bare-soil", 2:"buildings", 3:"cars", 4:"chaparral", 5:"court", 6:"dock", 7:"field", \
                        8:"grass", 9:"mobile-home", 10:"pavement", 11:"sand", 12:"sea", 13:"ship", 14:"tanks", 15:"trees", 16:"water"}
            self.map_transforms = Compose([
                                        Resize(size=args.size, interpolation=Image.NEAREST),
                                        CenterCrop(size=(args.size, args.size))
                                        ])
        else:
            self.labels = [name.split(" ")[2] for name in filenames]
        self.image_transforms = image_transforms
        self.root = root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.root is not None:
            img_path = os.path.join(self.root, str(self.images[idx]))
            if self.withsegmaps:
                map_path = os.path.join(self.root, str(self.segmaps[idx]))
        else:
            img_path = str(self.images[idx])
            if self.withsegmaps:
                map_path = str(self.segmaps[idx])
        image = self.image_transforms(Image.open(img_path))
        if self.withsegmaps:
            map = torch.tensor(np.array(self.map_transforms(Image.open(map_path))), dtype=torch.int64) - 1
            label_counts = {label: 0 for label in self.classes.keys()}
            for label in self.classes.keys():
                label_counts[label] = torch.sum(map == label).item()
            return image, map, label_counts, img_path
        else:
            label = self.labels[idx]
            return image, label, img_path

def find_additional_classes(label_counts):
    additional_classes_list = []

    for i, current_sample in enumerate(label_counts):
        current_classes = set(key for key, val in current_sample.items() if val > 0)
        additional_classes = set()

        for j, other_sample in enumerate(label_counts):
            if i != j:
                other_classes = set(key for key, val in other_sample.items() if val > 0)
                
                if current_classes.issubset(other_classes):
                    additional_classes.update(other_classes - current_classes)

        additional_classes_list.append(list(additional_classes))

    return additional_classes_list

def save_segmaps_dataset(dataloader, path_save):
    all_image_features, all_image_maps, all_label_counts, all_image_paths = [], [], [], []
    with torch.no_grad():
        ite = 128
        for images, maps, label_counts, img_paths in dataloader:
            print(ite, end='\r')
            images = images.cuda(0, non_blocking=True)
            image_features = model.encode_image(images)           
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
            all_image_features.append(image_features)
            all_image_maps.append(maps)
            for idx in range(len(img_paths)):
                all_image_paths.append(img_paths[idx])
                temp_label_counts = {}
                for class_label in label_counts.keys():
                    count = label_counts[class_label][idx].item()
                    temp_label_counts[class_label] = count
                all_label_counts.append(temp_label_counts)
            ite += 128
        print('Finding additional classes...')
        all_additional_classes = find_additional_classes(all_label_counts)
        all_image_features = torch.cat(all_image_features, dim=0)
        all_maps = torch.cat(all_image_maps, dim=0)
        dict_save = {}
        dict_save['feats'] = all_image_features.data.cpu().numpy()
        dict_save['maps'] = all_maps.data.cpu().numpy()
        dict_save['label_counts'] = all_label_counts
        dict_save['additional_classes'] = all_additional_classes
        dict_save['paths'] = all_image_paths
        directory = os.path.dirname(path_save)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path_save,"wb") as f:
            print('Writing picke file...')
            pickle.dump(dict_save,f)

def save_dataset(dataloader, path_save):
    all_image_features, all_labels, all_image_paths = [], [], []
    with torch.no_grad():
        ite = 128
        for images, labels, img_paths in dataloader:
            print(ite, end='\r')
            images = images.cuda(0, non_blocking=True)
            image_features = model.encode_image(images)           
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
            all_image_features.append(image_features)
            for idx in range(len(img_paths)):
                all_labels.append(labels[idx])
                all_image_paths.append(img_paths[idx])
            ite += 128
        all_image_features = torch.cat(all_image_features, dim=0)
        dict_save = {}
        dict_save['feats'] = all_image_features.data.cpu().numpy()
        dict_save['labels'] = all_labels
        dict_save['paths'] = all_image_paths
        directory = os.path.dirname(path_save)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path_save,"wb") as f:
            print('Writing picke file...')
            pickle.dump(dict_save,f)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Extracting features from the model')
    parser.add_argument('--model', type=str, default='clip', choices=['remoteclip', 'clip'], help='pre-trained model')
    parser.add_argument('--model_type', type=str, default='ViT-L-14', choices=['RN50', 'ViT-B-32', 'ViT-L-14'], help='pre-trained model type')
    parser.add_argument('--dataset', type=str, default='patternnet', choices=['dlrsd', 'patternnet', 'seasons'], help='choose dataset')
    parser.add_argument('--size', type=int, default=224, help='resize and crop size')
    parser.add_argument('--batch_size', type=int, default=128, help='dataloader batch size')
    args = parser.parse_args()

    model, _, preprocess_images = open_clip.create_model_and_transforms(args.model_type)
    tokenizer = open_clip.get_tokenizer(args.model_type)

    if args.model == 'remoteclip':
        ckpt = torch.load(f"models/RemoteCLIP-{args.model_type}.pt", map_location="cpu")
    elif args.model == 'clip':
        ckpt = torch.load(f"models/CLIP-{args.model_type}.bin", map_location="cpu")
    message = model.load_state_dict(ckpt)
    print(message)
    print(f"{args.model} {args.model_type} has been loaded!")
    model = model.cuda().eval()

    if args.dataset == 'dlrsd':
        full_dataset_path = "/mnt/datalv/bill/datasets/data/DLRSD/dlrsd.csv"
        full_dataset = ImageSegMapsDataset(full_dataset_path, image_transforms=preprocess_images, root='/mnt/datalv/bill/datasets/data/DLRSD/')
        full_dataloader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        save_segmaps_dataset(full_dataloader, '/mnt/datalv/bill/datasets/clip_features/dlrsd/dlrsd.pkl')
    elif args.dataset == 'patternnet':
        full_dataset_path = "/mnt/datalv/bill/datasets/data/PatternNet/patternnet.csv"
        full_dataset = ImageSegMapsDataset(full_dataset_path, image_transforms=preprocess_images, withsegmaps=False, root='/mnt/datalv/bill/datasets/data/PatternNet/')
        full_dataloader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        save_dataset(full_dataloader, '/mnt/datalv/bill/datasets/clip_features/patternnet/patternnet_clip.pkl')
    '''
    elif args.dataset == 'seasons':
        full_dataset_path = "./data/SSL4EO-S12/50k_images.csv"
        full_dataset = ImageDomainLabels(full_dataset_path, root='./data/SSL4EO-S12', transforms=preprocess_val)
        full_dataloader = DataLoader(full_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        save_dataset(full_dataloader, './clip_features/seasons/seasons.pkl')
    '''