import torch
import numpy as np
from PIL import Image
import csv
import open_clip

def load_model(model_name, model_type):
    model, _, preprocess_images = open_clip.create_model_and_transforms(model_type)
    tokenizer = open_clip.get_tokenizer(model_type)

    if model_name == 'remoteclip':
        ckpt = torch.load(f"models/RemoteCLIP-{model_type}.pt", map_location="cpu")
    elif model_name == 'clip':
        ckpt = torch.load(f"models/CLIP-{model_type}.bin", map_location="cpu")
    message = model.load_state_dict(ckpt)
    print(message)
    print(f"{model_name} {model_type} has been loaded!")
    model = model.cuda().eval()

    return model, preprocess_images, tokenizer

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

def create_metrics_final(at, methods):
    metrics_final = {method: {f"R@{k}": [] for k in at} for method in methods}
    for method in metrics_final:
        metrics_final[method].update({f"P@{k}": [] for k in at})
        metrics_final[method]["AP"] = []
    return metrics_final

def create_metrics_per_prompt(prompts, at, methods):
    metrics_per_prompt = {prompt: {method: {f"R@{k}": [] for k in at} for method in methods} for i in range(len(prompts)) for prompt in prompts[i]}
    for prompt in metrics_per_prompt:
        for method in metrics_per_prompt[prompt]:
            metrics_per_prompt[prompt][method].update({f"P@{k}": [] for k in at})
            metrics_per_prompt[prompt][method]["AP"] = []
    return metrics_per_prompt

# Class mapping for merging specific classes
class_mapping = {
    'denseresidential': 'residential',
    'sparseresidential': 'residential',
    'closedroad': 'road',
    'intersection': 'road',
    'bridge': 'roadpass',
    'overpass': 'roadpass',
    'ferryterminal': 'pier',
    'harbor': 'pier',
    'parkingspace': 'parking',
    'parkinglot': 'parking'
}

# Function to apply class mapping
def apply_class_mapping(label, class_mapping):
    return class_mapping.get(label, label)

# Function to fix specific attribute labels
def fix_query_attributelabels(attribute, query_attributelabels):
    if attribute == 'density':
        query_attributelabels = [x.replace('densitydenseresidential', 'densityresidential') for x in query_attributelabels]
        query_attributelabels = [x.replace('densitysparseresidential', 'densityresidential') for x in query_attributelabels]
    elif attribute == 'shape':
        query_attributelabels = [x.replace('shapeclosedroad', 'shaperoad') for x in query_attributelabels]
        query_attributelabels = [x.replace('shapeintersection', 'shaperoad') for x in query_attributelabels]
    elif attribute == 'context':
        query_attributelabels = [x.replace('contextbridge', 'contextroadpass') for x in query_attributelabels]
        query_attributelabels = [x.replace('contextoverpass', 'contextroadpass') for x in query_attributelabels]
    elif attribute == 'existence':
        query_attributelabels = [x.replace('existenceferryterminal', 'existencepier') for x in query_attributelabels]
        query_attributelabels = [x.replace('existenceharbor', 'existencepier') for x in query_attributelabels]
        query_attributelabels = [x.replace('existenceparkingspace', 'existenceparking') for x in query_attributelabels]
        query_attributelabels = [x.replace('existenceparkinglot', 'existenceparking') for x in query_attributelabels]
    return query_attributelabels