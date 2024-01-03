import os
import time
import pickle
import numpy as np
import torch
import open_clip
import argparse
from PIL import Image

from utils import *


def read_segmaps_dataset_features(pickle_dir):
    with open(pickle_dir, 'rb') as f:
        data = pickle.load(f)
    all_image_features = torch.from_numpy(data['feats'].astype("float32")).float().to('cuda')
    all_maps = data['maps']
    all_label_counts = data['label_counts']
    all_additional_classes = data['additional_classes']
    all_paths = data['path']
    return all_image_features, all_maps, all_label_counts, all_additional_classes, all_paths

def metrics_calc(rankings, cls, label_counts, classes_change, query_index, at):
    metrics = {}
    query_classes = set(key for key, val in label_counts[query_index].items() if val > 0)
    cls_id = list(classes_change.keys())[list(classes_change.values()).index(cls)]
    query_classes.add(cls_id)  # Add the additional class

    relevant_ranking_indices = [idx for idx in range(len(rankings)) if query_classes.issubset(set(key for key, val in label_counts[rankings[idx]].items() if val > 0))]

    precisions = []
    for idx, rank in enumerate(relevant_ranking_indices, start=1):
        precision_at_rank = idx / (rank + 1)  # rank is zero-indexed
        precisions.append(precision_at_rank)

    ap = sum(precisions) / len(precisions) if precisions else 0
    metrics["AP"] = round(ap*100, 2)

    for k in at:
        top_k_indices = rankings[:k]

        # Count how many retrieved items are relevant
        relevant_count = sum(idx in relevant_ranking_indices for idx in top_k_indices)

        # Calculate Recall@k and Precision@k
        recall_at_k = relevant_count / len(query_classes) if query_classes else 0
        precision_at_k = relevant_count / k if k else 0

        metrics[f"R@{k}"] = round(recall_at_k*100, 2)
        metrics[f"P@{k}"] = round(precision_at_k*100, 2)

    return metrics

def calculate_rankings(method, query_features, text_features, database_features, lam=0.5):

    if np.array([x in method for x in ['Image','Add Similarities','Multiply Similarities', 'Minimum Similarity']]).any():
        sim_img = (query_features @ database_features.t())
    if np.array([x in method for x in ['Text','Add Similarities','Multiply Similarities', 'Minimum Similarity']]).any():    
        sim_text = (text_features @ database_features.t())

    if "norm" in method.lower():
        sim_img = norm_cdf(sim_img)
        sim_text = norm_cdf(sim_text)

    if "image only" in method.lower():
        ranks = torch.argsort(sim_img, descending=True)
    elif "text only" in method.lower():
        ranks = torch.argsort(sim_text, descending=True)
    elif "add similarities" in method.lower():
        ranks = torch.argsort(sim_img + sim_text, descending=True)
    elif "multiply similarities" in method.lower():
        ranks = torch.argsort(torch.mul(sim_img, sim_text), descending=True)
    elif "minimum similarity" in method.lower():
        ranks = torch.argsort(torch.maximum(sim_img, sim_text), descending=False)
    
    return ranks.detach().cpu()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Validating extracted features')
    parser.add_argument('--model_name', type=str, default='ViT-L-14', choices=['RN50', 'ViT-B-32', 'ViT-L-14'], help='pre-trained model to use')
    parser.add_argument('--dataset', type=str, default='dlrsd', choices=['dlrsd', 'patternet', 'seasons'], help='choose dataset')
    parser.add_argument('--size', type=int, default=224, help='resize and crop size')
    parser.add_argument('--lr', type=float, default=0.01, help="lr")
    parser.add_argument('--iters', type=int, default=10, help="opt iters")
    parser.add_argument('--alpha', type=float, default=0.8, help="ema rate")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay")
    parser.add_argument('--init', type=str, default="knn", help="initialize token emb", choices=["rnd", "knn"])
    args = parser.parse_args()
    
    model, _, _ = open_clip.create_model_and_transforms(args.model_name)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    ckpt = torch.load(f"models/RemoteCLIP-{args.model_name}.pt", map_location="cpu")
    message = model.load_state_dict(ckpt)
    print(message)
    print(f"{args.model_name} has been loaded!")

    model = model.cuda().eval()

    root = "/mnt/datalv/bill/datasets/"
    #methods = ["Add Similarities Norm"]

    methods = ["Image only", "Text only", "Add Similarities", "Multiply Similarities", "Add Similarities Norm", "Multiply Similarities Norm"]

    #metrics = {}
    #for method in methods:
    #    metrics[method] = {}
    if args.dataset == 'dlrsd':
        print('Reading features and maps...')
        features, maps, label_counts, additional_classes, paths = read_segmaps_dataset_features(root+'/clip_features/dlrsd/dlrsd.pkl')
        print('Features and maps are loaded!')
        classes_change = {0: "airplane", 1: "bare-soil", 2:"buildings", 3:"cars", 4:"chaparral", 5:"court", 6:"dock", 7:"field", \
                        8:"grass", 9:"mobile-home", 10:"pavement", 11:"sand", 12:"sea", 13:"ship", 14:"tanks", 15:"trees", 16:"water"}
        at = [5, 10, 20, 50]
        
    add_classes = replace_class_names(additional_classes, classes_change)

    metrics_final = {method: {f"R@{k}": [] for k in at} for method in methods}
    for method in metrics_final:
        metrics_final[method].update({f"P@{k}": [] for k in at})
        metrics_final[method]["AP"] = []
    
    for i in range(len(paths)):
        print(f'Retrieval running for query {i}', end='\r')
        query_feature = features[i]
        for j, cls in enumerate(add_classes[i]):
            prompt = "with " + cls
            text = tokenizer(prompt).to('cuda')
            text_feature = model.encode_text(text)
            text_feature = (text_feature / text_feature.norm(dim=-1, keepdim=True)).squeeze().detach().to(torch.float32)
            for method in methods:
                time1 = time.time()
                rankings = calculate_rankings(method, query_feature, text_feature, features)
                temp_metrics = metrics_calc(rankings, cls, label_counts, classes_change, i, at)
                
                # Accumulate metrics for each method
                for k in at:
                    metrics_final[method][f"R@{k}"].append(temp_metrics[f"R@{k}"])
                    metrics_final[method][f"P@{k}"].append(temp_metrics[f"P@{k}"])
                metrics_final[method]["AP"].append(temp_metrics["AP"])

    # Calculate the average for each metric
    for method in metrics_final:
        for metric in metrics_final[method]:
            metrics_final[method][metric] = round(sum(metrics_final[method][metric]) / len(metrics_final[method][metric]) if metrics_final[method][metric] else 0, 2)

    print(metrics_final)

    dict_to_csv(metrics_final, args.dataset + 'metrics.csv') #time.strftime("%Y_%m_%d_%H_%M_%S")+'.csv')
