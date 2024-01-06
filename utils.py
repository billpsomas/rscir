import torch
import numpy as np
from PIL import Image
import csv

def replace_class_names(additional_classes, classes_change):
    named_classes_list = []

    for sublist in additional_classes:
        named_sublist = [classes_change[item] for item in sublist]
        named_classes_list.append(named_sublist)

    return named_classes_list

def norm_cdf(matrix, dim=0):
    row_means = matrix.mean(dim=dim, keepdim=True)
    row_stds = matrix.std(dim=dim, keepdim=True)
    matrix = 0.5 * (1 + torch.erf((matrix - row_means) / (row_stds * torch.sqrt(torch.tensor(2.0)))))
    return matrix

def norm_gaussian(matrix, dim=0):
    row_means = matrix.mean(dim=dim, keepdim=True)
    row_stds = matrix.std(dim=dim, keepdim=True)
    matrix = (matrix-row_means)/row_stds
    return matrix

def calculate_ranks(input_tensor):
    num_rows, num_cols = input_tensor.shape
    ranked_tensor = torch.zeros_like(input_tensor, dtype=torch.int64).to('cuda')
    for i in range(num_rows):
        row = input_tensor[i]
        sorted_indices = torch.argsort(row, descending=True)
        ranked_tensor[i, sorted_indices] = torch.arange(1, num_cols + 1).to('cuda')
    return(ranked_tensor)

def preprocess_image(img_path, new_width, new_height, crop=False):
    pil_img = Image.open(img_path).convert("RGB")
    width, height = pil_img.size

    if crop:
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        pil_img = pil_img.crop((left, top, right, bottom))
    else:
        pil_img = pil_img.resize((new_width, new_height))
    img_array = np.array(pil_img)
    
    return img_array

def dict_to_csv(metrics_dict, filename):
    # Extract headers for CSV (metric names)
    headers = ["Method"] + list(next(iter(metrics_dict.values())).keys())

    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        writer.writeheader()
        for method, metrics in metrics_dict.items():
            row = {'Method': method}
            row.update(metrics)
            writer.writerow(row)

def timer(start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))