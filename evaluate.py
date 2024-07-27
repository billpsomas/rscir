import os
import time
import pickle
import numpy as np
import torch
import open_clip
import argparse
from PIL import Image
import re
from collections import defaultdict
from tqdm import tqdm

from utils import *


# Function to read features from a pickle file
def read_dataset_features(pickle_dir):
    with open(pickle_dir, 'rb') as f:
        data = pickle.load(f)
    all_image_features = torch.from_numpy(data['feats'].astype("float32")).float().to('cuda')
    all_labels = [label.replace('_', '') for label in data['labels']]
    all_paths = data['paths']
    return all_image_features, all_labels, all_paths

# Function to read CSV files
def read_csv(file_path):
    image_filenames = []
    attributes = []
    attribute_values = []
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            image_filenames.append(row[0])
            attributes.append(row[1])
            attribute_values.append(row[2])
    return image_filenames, attributes, attribute_values

# Function to find relative indices of query paths in the dataset paths
def find_relative_indices(query_paths, paths):
    path_index_map = {os.path.basename(path): i for i, path in enumerate(paths)}
    relative_indices = []
    for query_path in query_paths:
        index = path_index_map.get(query_path)
        if index is not None:
            relative_indices.append(index)
    return relative_indices

# Function to create prompts for attribute evaluation
def create_prompts(paired):
    # Store attributes for each category
    category_to_attributes = defaultdict(set)
    for category, attribute in paired:
        category_to_attributes[category].add(attribute)
    prompts = []
    for category, attribute in paired:
        # Get all other attributes for this category
        other_attributes = category_to_attributes[category] - {attribute}
        # Add each other attribute to the new list
        # If there are multiple other attributes, add them all
        other_attributes = list(other_attributes)
        prompts.append(other_attributes)
    return prompts

# Function to calculate various metrics for the retrieved results
def metrics_calc(rankings, prompt, paths, filename_to_index_map, attribute_values, at, query_class, query_labels):
    metrics = {}
    # Convert rankings to filenames to find their corresponding attribute values
    retrieved_filenames = [os.path.basename(paths[idx]) for idx in rankings]
    # Find indices in query_filenames using the precomputed map
    retrieved_indices = [filename_to_index_map.get(filename, -1) for filename in retrieved_filenames]
    # Map original query class -> our query class
    query_class_mapped = apply_class_mapping(query_class, class_mapping)
    
    # Determine if each retrieval is relevant (True or False)
    # Take into account both class (has to be the same as query image) 
    # and attribute (has to be the same as query text - prompt)
    is_relevant = [attribute_values[idx] == prompt and apply_class_mapping(query_labels[idx], class_mapping) == query_class_mapped if idx != -1 else False for idx in retrieved_indices]

    # Calculate Average Precision (AP)
    precisions = []
    relevant_count = 0
    for k, rel in enumerate(is_relevant, start=1):
        if rel:
            relevant_count += 1
            precisions.append(relevant_count / k)

    ap = sum(precisions) / len(precisions) if precisions else 0
    metrics["AP"] = round(ap * 100, 2)

    # Calculate Precision@k and Recall@k
    total_relevant = sum(is_relevant)
    for k in at:
        relevant_at_k = sum(is_relevant[:k])
        precision_at_k = relevant_at_k / k if k else 0
        recall_at_k = relevant_at_k / total_relevant if total_relevant else 0

        metrics[f"P@{k}"] = round(precision_at_k * 100, 2)
        metrics[f"R@{k}"] = round(recall_at_k * 100, 2)

    return metrics

# Function to calculate rankings based on the selected method
def calculate_rankings(method, query_features, text_features, database_features, lam=0.5):

    if np.array([x in method for x in ['Image', 'Average Similarities', 'Weighted Similarities', 'Add Similarities', 'Multiply Similarities', 'Minimum Similarity']]).any():
        sim_img = (query_features @ database_features.t())
    if np.array([x in method for x in ['Text', 'Average Similarities', 'Weighted Similarities', 'Add Similarities', 'Multiply Similarities', 'Minimum Similarity']]).any():    
        sim_text = (text_features @ database_features.t())

    if "norm" in method.lower():
        sim_img = norm_cdf(sim_img)
        sim_text = norm_cdf(sim_text)

    if "image only" in method.lower():
        ranks = torch.argsort(sim_img, descending=True)
    elif "text only" in method.lower():
        ranks = torch.argsort(sim_text, descending=True)
    elif "average similarities" in method.lower():
        ranks = torch.argsort((sim_img + sim_text)/2, descending=True)
    elif "weighted similarities" in method.lower():
         ranks = torch.argsort((1-lam)*sim_img + lam*sim_text, descending=True)
    elif "add similarities" in method.lower():
        ranks = torch.argsort(sim_img + sim_text, descending=True)
    elif "multiply similarities" in method.lower():
        ranks = torch.argsort(torch.mul(sim_img, sim_text), descending=True)
    elif "minimum similarity" in method.lower():
        ranks = torch.argsort(torch.maximum(sim_img, sim_text), descending=False)
    
    return ranks.detach().cpu()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluating extracted features for Remote Sensing Composed Image Retrieval.')
    parser.add_argument('--model_name', type=str, default='remoteclip', choices=['remoteclip', 'clip'], help='pre-trained model')
    parser.add_argument('--model_type', type=str, default='ViT-L-14', choices=['RN50', 'ViT-B-32', 'ViT-L-14'], help='pre-trained model type')
    parser.add_argument('--dataset', type=str, default='patternnet', choices=['dlrsd', 'patternnet', 'seasons'], help='choose dataset')
    parser.add_argument('--attributes', nargs='+', default=['color', 'shape', 'density', 'quantity', 'context', 'existence'], choices=['color', 'shape', 'density', 'quantity', 'context', 'existence'], help='a list of attributes')
    parser.add_argument('--dataset_path', type=str, default='/mnt/datalv/bill/datasets/data/PatternNet/', help='PatternNet dataset path')
    parser.add_argument('--methods', nargs='+', default=["Weighted Similarities Norm"], choices=["Image only", "Text only", "Average Similarities", "Weighted Similarities Norm"], help='methods to evaluate')
    parser.add_argument('--lambdas', type=str, default='0.5', help='comma-separated list of lambda values')
    args = parser.parse_args()

    # Convert lambdas argument to a list of floats
    lambdas = list(map(float, args.lambdas.split(',')))

    # Load model and tokenizer
    model, _, tokenizer = load_model(args.model_name, args.model_type)

    # Read features from the specified dataset
    if args.dataset == 'patternnet':
        print('Reading features...')
        features, labels, paths = read_dataset_features(os.path.join(args.dataset_path, 'features', f'patternnet_{args.model_name}.pkl'))
        print('Features are loaded!')
        at = [5, 10, 15, 20]

    # Initialize metrics storage
    metrics_final = create_metrics_final(at, args.methods)

    if args.dataset == 'patternnet':
        for lam in lambdas:
            for attribute in args.attributes:
                metrics_final = create_metrics_final(at, args.methods)
                start = time.time()
                
                # Read query data from CSV file
                query_filenames, attributes, attribute_values = read_csv(os.path.join(args.dataset_path, 'PatternCom', f'{attribute}.csv'))
                query_labels = [re.split(r'\d', path)[0] for path in query_filenames] # or something like labels[relative_indices], should give the same
                
                # Fix query attribute labels
                query_attributelabels = [x + query_labels[ii] for ii, x in enumerate(attributes)]
                query_attributelabels = fix_query_attributelabels(attribute, query_attributelabels)

                # Pair attribute labels with attribute values
                paired = list(zip(query_attributelabels, attribute_values))

                # Create prompts based on paired data
                prompts = create_prompts(paired)
                relative_indices = find_relative_indices(query_filenames, paths)
                filename_to_index_map = {filename: i for i, filename in enumerate(query_filenames)}

                # Cache text features
                text_feature_cache = {}
                for i, idx in enumerate(tqdm(relative_indices, desc="Processing queries")):
                    query_feature = features[idx]
                    query_class = query_labels[i]  # Get the original class of the query image
                    for prompt in tqdm(prompts[i], desc="Processing prompts", leave=False):
                        # Check if the text feature for this prompt is already computed
                        if prompt not in text_feature_cache:
                            # If not, compute and cache it
                            text = tokenizer(prompt).to('cuda')
                            text_feature = model.encode_text(text)
                            text_feature = (text_feature / text_feature.norm(dim=-1, keepdim=True)).squeeze().detach().to(torch.float32)
                            text_feature_cache[prompt] = text_feature
                        else:
                            # If already computed, retrieve from cache
                            text_feature = text_feature_cache[prompt]
                        for method in args.methods:
                            rankings = calculate_rankings(method, query_feature, text_feature, features, lam)
                            temp_metrics = metrics_calc(rankings, prompt, paths, filename_to_index_map, attribute_values, at, query_class, query_labels)

                            # Accumulate metrics for each method
                            for k in at:
                                metrics_final[method][f"R@{k}"].append(temp_metrics[f"R@{k}"])
                                metrics_final[method][f"P@{k}"].append(temp_metrics[f"P@{k}"])
                            metrics_final[method]["AP"].append(temp_metrics["AP"])

                # Calculate average metrics
                for method in metrics_final:
                    for metric in metrics_final[method]:
                        metrics_final[method][metric] = round(sum(metrics_final[method][metric]) / len(metrics_final[method][metric]) if metrics_final[method][metric] else 0, 2)

                print(metrics_final)
                end = time.time()
                timer(start, end)

                # Save metrics to CSV file
                print('Writing results to CSV file...')
                results_dir = 'results'
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                results_file_path = os.path.join(results_dir, f'{args.dataset}_metrics_{args.model_name}_lambda{lam}_{attribute}.csv')
                dict_to_csv(metrics_final, results_file_path)