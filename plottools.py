import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
import os
from tqdm import tqdm

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"

#PLOT--FUNCTIONS----------------------------------------------------------------------------------------------------

def epsilon_vs_accuracy(results_list, epsilons, target = 0, autotarget = True):
    res_orig = []

    for i in range(len(epsilons)):
        res = results_list[i] 
        res_pert = []

        for i, res_dict in enumerate(res):
            accuracys_pert = []
            if autotarget == True:
                max_key_original = max(res_dict['original'], key=res_dict['original'].get) #original max value
                perturbed_target = res_dict['perturbed'][max_key_original]
            else:
                perturbed_target = res_dict['perturbed'][target]
            accuracys_pert.append(perturbed_target)
            res_pert.append(accuracys_pert)
        res_orig.append(res_pert)
    res_orig = np.array(res_orig)

    for image_index in range(res_orig.shape[1]):
        image_results = res_orig[:, image_index, 0]
        plt.plot(epsilons, image_results, label=f'Img {image_index+1}')

    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    if autotarget == True:
        plt.title('Accuracy of Target (autotarget) in Perturbed Image')
    else:
        plt.title(f'Accuracy of Target: {target} in Perturbed Image')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_top_results_ensamble(results, n_classes):
    for i, res_dict in enumerate(results):
        print(f'{i + 1}° Result')

        orig_classes = list(res_dict['original'].keys())
        orig_probabilities = list(res_dict['original'].values())
        orig_classes_p = list(res_dict['perturbed'].keys())
        orig_probabilities_p = list(res_dict['perturbed'].values())

        coarse_classes = list(res_dict['coarse_scores'].keys())
        coarse_probabilities = list(res_dict['coarse_scores'].values())
        coarse_classes_p = list(res_dict['pert_coarse_scores'].keys())
        coarse_probabilities_p = list(res_dict['pert_coarse_scores'].values())

        # Generate colors for bars

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(orig_classes[:n_classes], orig_probabilities[:n_classes])
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Probability')
        axes[0].set_title(f'Original Results - {i + 1}')

        axes[0].set_xticks(range(len(orig_classes[:n_classes])))
        axes[0].set_xticklabels(orig_classes[:n_classes], rotation=45, ha='right')
        axes[0].set_ylim(0, 1)  # Set y-axis limit from 0 to 1

        # Plot probability bar chart
        axes[1].bar(orig_classes_p[:n_classes], orig_probabilities_p[:n_classes])
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Probability')
        axes[1].set_title(f'Original Results Perturbed - {i + 1}')
    
        axes[1].set_xticks(range(len(orig_classes_p[:n_classes])))
        axes[1].set_xticklabels(orig_classes_p[:n_classes], rotation=45, ha='right')
        axes[1].set_ylim(0, 1)  # Set y-axis limit from 0 to 1
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(coarse_classes[:n_classes], coarse_probabilities[:n_classes])
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Probability')
        axes[0].set_title(f'Coarse Results - {i + 1}')

        axes[0].set_xticks(range(len(coarse_classes[:n_classes])))
        axes[0].set_xticklabels(coarse_classes[:n_classes], rotation=45, ha='right')
        axes[0].set_ylim(0, 1)  # Set y-axis limit from 0 to 1

        # Plot probability bar chart
        axes[1].bar(coarse_classes_p[:n_classes], coarse_probabilities_p[:n_classes])
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Probability')
        axes[1].set_title(f'Coarse Results Perturbed - {i + 1}')
    
        axes[1].set_xticks(range(len(coarse_classes_p[:n_classes])))
        axes[1].set_xticklabels(coarse_classes_p[:n_classes], rotation=45, ha='right')
        axes[1].set_ylim(0, 1)  # Set y-axis limit from 0 to 1
        plt.tight_layout()
        plt.show()

def plot_results_ensamble_target_vs_original(results, target):
    for i, res_dict in enumerate(results):
        print(f'{i + 1}° Result')

        # Target original - Target perturbed
        orig_target = res_dict['coarse_scores'][target]
        perturbed_target = res_dict['pert_coarse_scores'][target]

        # Max original - Max perturbed
        max_key_original = max(res_dict['coarse_scores'], key=res_dict['coarse_scores'].get)
        max_value_original = res_dict['coarse_scores'][max_key_original]
        max_key_perturbed = max(res_dict['pert_coarse_scores'], key=res_dict['pert_coarse_scores'].get)
        max_value_perturbed = res_dict['pert_coarse_scores'][max_key_perturbed]

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot target bar chart
        bar_width = 0.35
        indices = np.arange(2)

        bars1 = axes[0].bar(indices, [orig_target, perturbed_target], width=bar_width)
        axes[0].set_ylabel('Probability')
        axes[0].set_title(f'Original vs Perturbed - Target: {target}')
        axes[0].set_xticks(indices)
        axes[0].set_xticklabels(['original', 'perturbed'], rotation=45, ha='right')
        axes[0].set_ylim(0, 1)  # Set y-axis limit from 0 to 1

        # Add labels indicating the height of the bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].annotate('%.2f' % height,
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')

        # Plot probability bar chart
        bars2 = axes[1].bar(indices, [max_value_original, max_value_perturbed], width=bar_width)
        axes[1].set_ylabel('Probability')
        axes[1].set_title(f'Original vs Perturbed - Maximum')
        axes[1].set_xticks(indices)
        axes[1].set_xticklabels([max_key_original, max_key_perturbed], rotation=45, ha='right')
        axes[1].set_ylim(0, 1)  # Set y-axis limit from 0 to 1

        # Add labels indicating the height of the bars
        for bar in bars2:
            height = bar.get_height()
            axes[1].annotate('%.3f' % height,
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

def plot_results_ensamble_target_vs_original_default_labels(results, target):
    for i, res_dict in enumerate(results):
        print(f'{i + 1}° Result')

        # Target original - Target perturbed
        orig_target = res_dict['original'][target]
        perturbed_target = res_dict['perturbed'][target]

        # Max original - Max perturbed
        max_key_original = max(res_dict['original'], key=res_dict['original'].get)
        max_value_original = res_dict['original'][max_key_original]
        max_key_perturbed = max(res_dict['perturbed'], key=res_dict['perturbed'].get)
        max_value_perturbed = res_dict['perturbed'][max_key_perturbed]

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot target bar chart
        bar_width = 0.35
        indices = np.arange(2)

        bars1 = axes[0].bar(indices, [orig_target, perturbed_target], width=bar_width)
        axes[0].set_ylabel('Probability')
        axes[0].set_title(f'Original vs Perturbed - Target: {target}')
        axes[0].set_xticks(indices)
        axes[0].set_xticklabels(['original', 'perturbed'], rotation=45, ha='right')
        axes[0].set_ylim(0, 1)  # Set y-axis limit from 0 to 1

        # Add labels indicating the height of the bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].annotate('%.2f' % height,
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')

        # Plot probability bar chart
        bars2 = axes[1].bar(indices, [max_value_original, max_value_perturbed], width=bar_width)
        axes[1].set_ylabel('Probability')
        axes[1].set_title(f'Original vs Perturbed - Maximum')
        axes[1].set_xticks(indices)
        axes[1].set_xticklabels([max_key_original, max_key_perturbed], rotation=45, ha='right')
        axes[1].set_ylim(0, 1)  # Set y-axis limit from 0 to 1

        # Add labels indicating the height of the bars
        for bar in bars2:
            height = bar.get_height()
            axes[1].annotate('%.3f' % height,
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

def plot_bar_results(results, target, print_vals = False):
    num_results = len(results)
    # Initialize arrays to store data for bar plots
    orig_targets = np.zeros(num_results)
    perturbed_targets = np.zeros(num_results)
    max_values_original_0 = np.zeros(num_results)
    max_values_perturbed_0 = np.zeros(num_results)
    max_values_original_1 = np.zeros(num_results)
    max_values_perturbed_1 = np.zeros(num_results)

    for i, res_dict in enumerate(results):
        # Target original - Target perturbed
        orig_targets[i] = res_dict['original'][target]
        perturbed_targets[i] = res_dict['perturbed'][target]

        # Max original in both original and perturbed
        max_key_original = max(res_dict['original'], key=res_dict['original'].get)
        max_values_original_0[i] = res_dict['original'][max_key_original]
        max_values_perturbed_0[i] = res_dict['perturbed'][max_key_original]

        # Max perturbed in both original and perturbed
        max_key_perturbed = max(res_dict['perturbed'], key=res_dict['perturbed'].get)
        max_values_original_1[i] = res_dict['original'][max_key_perturbed]
        max_values_perturbed_1[i] = res_dict['perturbed'][max_key_perturbed]
        if print_vals == True:
            print(f'Max in {i + 1}° Img: original {max_key_original} - perturbed {max_key_perturbed}\n')


    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    # Bar plot for Target
    indices = np.arange(1, num_results + 1)
    bar_width = 0.35

    axes[0].bar(indices - bar_width/2, orig_targets, width=bar_width, label='original')
    axes[0].bar(indices + bar_width/2, perturbed_targets, width=bar_width, label='perturbed')
    axes[0].set_xlabel('Result')
    axes[0].set_ylabel('Probability')
    axes[0].set_title(f'Target: {target}')
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels([f'Img {i+1}' for i in range(num_results)])
    axes[0].legend()

    axes[1].bar(indices - bar_width/2, max_values_original_0, width=bar_width, label='original')
    axes[1].bar(indices + bar_width/2, max_values_perturbed_0, width=bar_width, label='perturbed')
    axes[1].set_xlabel('Result')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Original Maximum')
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels([f'Img {i+1}' for i in range(num_results)])
    axes[1].legend()

    axes[2].bar(indices - bar_width/2, max_values_original_1, width=bar_width, label='original')
    axes[2].bar(indices + bar_width/2, max_values_perturbed_1, width=bar_width, label='perturbed')
    axes[2].set_xlabel('Result')
    axes[2].set_ylabel('Probability')
    axes[2].set_title('Perturbed Maximum')
    axes[2].set_xticks(indices)
    axes[2].set_xticklabels([f'Img {i+1}' for i in range(num_results)])
    axes[2].legend()

    plt.tight_layout()
    plt.show()
