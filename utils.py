import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
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


#PRE-PROCESSING----------------------------------------------------
def get_imagenet_classes():
    try:
        with open(r"C:\Users\Usuario\Documents\Trini\Facultad\Tesis\Código Personal\imagenet_classes.txt") as f:
            classes = [line.strip() for line in f.readlines()]
            return classes
    except FileNotFoundError:
        print(f"File not found, try modifying the path in utils.py")
        return None

def get_default_weights():
    weights = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return weights

def get_fine2course_mapping():
    classes = get_imagenet_classes()
    coarse_classes = classes.copy()

    #Falta chequear bien en detalle y ver una forma más optima de hacerlo
    for i in np.arange(151, 269):
        coarse_classes[i] = 'dog'
    for i in np.arange(281, 286):
        coarse_classes[i] = 'cat'
    for i in np.arange(7, 51):
        coarse_classes[i] = 'bird'
    
    fine_to_coarse_mapping = {}
    # Read fine classes from the file and map them to coarse classes
    for i, line in enumerate(classes):
        fine_class = line.strip()  # Remove leading/trailing whitespace
        # Choose a coarse class randomly from the list
        fine_to_coarse_mapping[fine_class] = coarse_classes[i]
    
    unique_coarse = []
    [unique_coarse.append(value) for value in fine_to_coarse_mapping.values() if value not in unique_coarse]

    return fine_to_coarse_mapping, unique_coarse
    

use_cuda=False
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

#CLASSIFING----------------------------------------------------------

#folder_path: de la carpeta con imagenes
#model_name: nombre del modelo 
#classes: en general las de Imagenet (el txt está en internet)
#top_classes: número de clases para guardar (de mayor a menor en resultados)

def get_class_model_names():
    return ['resnet18', 'resnet50', 'efficientnetB0']

def classify_images(folder_path, model_name, classes, top_classes=3):
    # Load the pretrained model
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "efficientnetB0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        raise ValueError("Unsupported model name.")

    # Set the model to evaluation mode
    model.eval()

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # List to store results
    results = []

    # Iterate through images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Assuming images are either jpg or png
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image_tensor = transform(image).unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                outputs = torch.softmax(model(image_tensor), dim=1)

            # Get top classes
            top_probabilities, top_indices = torch.topk(outputs, top_classes)
            top_probabilities = top_probabilities.squeeze().tolist()
            top_indices = top_indices.squeeze().tolist()

            # Map indices to class labels
            idx_to_class = {idx: classes[idx] for idx in range(len(classes))}
            class_probabilities = {idx_to_class[idx]: prob for idx, prob in zip(top_indices, top_probabilities)}

            # Append result to list
            results.append(class_probabilities)

    return results

#Ploting results
def plot_results(folder_path, results, adv_attack = False, coarse_classes = False, number = 3):
    # Get list of image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]

    # Check if number of images matches number of results
    if len(image_files) != len(results):
        raise ValueError("Number of images in folder does not match number of results.")

    for image_file, res_dict in zip(image_files, results):
        
        if adv_attack and coarse_classes == True:
            classes = list(res_dict['pert_coarse_scores'].keys())[0:number]
            probabilities = list(res_dict['pert_coarse_scores'].values())[0:number]
        elif adv_attack == True and coarse_classes == False:
            classes = list(res_dict['perturbed'].keys())[0:number]
            probabilities = list(res_dict['perturbed'].values())[0:number]
        else:
            classes = list(res_dict.keys())[0:number]
            probabilities = list(res_dict.values())[0:number]
        image_path = os.path.join(folder_path, image_file)

        # Generate colors for bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot image
        image = Image.open(image_path)
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title('Image')

        # Plot probability bar chart
        axes[1].bar(classes, probabilities, color=colors)
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Probability')
        axes[1].set_title('Probabilities')
    
        axes[1].set_xticks(range(len(classes)))
        axes[1].set_xticklabels(classes, rotation=45, ha='right')
        axes[1].set_ylim(0, 1)  # Set y-axis limit from 0 to 1

        plt.tight_layout()
        plt.show()


#ATTACKING------------------------------------------------------------
        
def denorm(image_tensor, mean, std):
    """Denormalize an image tensor."""
    for t, m, s in zip(image_tensor, mean, std):
        t.mul_(s).add_(m)
    return image_tensor

def ifgsm_attack(input, epsilon, data_grad):
    iter = int(min([epsilon + 4, epsilon * 1.25]))  # Number of iterations
    
    alpha = 1
    pert_out = input.clone().detach()
    
    for i in range(iter):
        pert_out = pert_out + (alpha / 255) * data_grad.sign()

        if torch.norm((pert_out - input), p=float('inf')) > epsilon / 255:
            break
    
    pert_out = torch.clamp(pert_out, 0, 1)
    adv_pert = torch.clamp(pert_out - input, 0,1)

    return pert_out, adv_pert

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image, epsilon*sign_data_grad

#1) ------------------SIMPLE ATTACK iFGSM------------------

def simple_attack(image_folder, model, weights, epsilon, classes, targeted = False, t = 0, num_classes=3, graph=False, folder=False, control=False):
    """Test function to generate adversarial images and obtain predictions."""
    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Prepare output directory
    if folder:
        if targeted == False:
            output_dir = f"simple-attack_epsilon-{epsilon}-untargeted"
        else: 
            output_dir = f"simple-attack_epsilon-{epsilon}-targeted-{classes[t]}"
        os.makedirs(output_dir, exist_ok=True)

    results = []
    print('Iterating over images in folder')

    for image_name in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)

        # Load and transform image
        img = Image.open(image_path)
        transformed_image = weights(img).unsqueeze(0)
        original_image = transformed_image.clone()

        # Calculate loss
        transformed_image.requires_grad = True
        output = model(normalize(transformed_image))
        loss = nn.CrossEntropyLoss()

        if targeted == True:
            target = torch.tensor([t])
            cost = -loss(output, target) #minimizar el error en la direc del target
        else:
            target = output.max(1)[1]
            cost = -loss(output, target) #minimizar el error en la direc del target
#        target = output.max(1)[1]
#        loss = F.nll_loss(output, target)

        model.zero_grad()
#        loss.backward()
        grad = torch.autograd.grad(cost, transformed_image)[0]

        # Generate adversarial image
        perturbed_image, adv_pert = ifgsm_attack(transformed_image, epsilon, grad)

        # Get predictions
        original_output = model(normalize(original_image))
        original_probs = torch.softmax(original_output, dim=1)
        perturbed_output = model(normalize(perturbed_image))
        perturbed_probs = torch.softmax(perturbed_output, dim=1)

        # Sort predictions
        original_top_classes = original_probs[0].topk(num_classes)
        perturbed_top_classes = perturbed_probs[0].topk(num_classes)

        original_dict = {classes[idx.item()]: prob.item() for idx, prob in zip(original_top_classes.indices, original_top_classes.values)}
        perturbed_dict = {classes[idx.item()]: prob.item() for idx, prob in zip(perturbed_top_classes.indices, perturbed_top_classes.values)}

        results.append({'original': original_dict, 'perturbed': perturbed_dict})

        # Save adversarial image
        if folder:
            perturbed_image = perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            perturbed_image = (perturbed_image * 255).astype('uint8')
            Image.fromarray(perturbed_image).save(os.path.join(output_dir, image_name))

        # Plot if graph is True
        if graph:
            print(f'Epsilon: {epsilon}')
            if targeted == True:
                print(f'Targeted Attack = {classes[t]}')
            else:
                print(f'Untargeted Attack. Maximize Original Output')

            plt.subplot(1, 3, 1)
            plt.imshow(original_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Original: {max(original_dict, key=original_dict.get)} - {np.round(max(original_dict.values()), 3)}",  fontsize=7)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            if folder:
                plt.imshow(perturbed_image)
            else:
                plt.imshow(perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))            
            plt.title(f"Perturbed: {max(perturbed_dict, key=perturbed_dict.get)} - {np.round(max(perturbed_dict.values()), 3)}", fontsize=7)
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(adv_pert.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Perturbation", fontsize=7)
            plt.axis('off')
            
            plt.subplots_adjust(wspace=0.5)  # Add space between the plots
            plt.show()

    # Save results to txt
    if folder:
        with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
#            f.write(f"Epsilon used: {epsilon}\n")
            for result in results:
                f.write(f"Original: {result['original']}\nPerturbed: {result['perturbed']}\n\n")

    return results


#2)-----------------ENSAMBLE ATTACK iFGSM-------------------------
def ensamble_attack(image_folder, models, weights, epsilon, classes, targeted = False, t = 0, num_classes=3, graph=False, folder=False, control=False):
    """Test function to generate adversarial images and obtain predictions using an ensemble of models."""
    for model in models:
        model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Prepare output directory
    if folder:
        if targeted == False:
            output_dir = f"ensamble-attack_epsilon-{epsilon}-untargeted"
        else:
            output_dir = f"ensamble-attack_epsilon-{epsilon}-targeted"
        os.makedirs(output_dir, exist_ok=True)

    results = []
    print('Iterating over images in folder')
    for image_name in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)

        # Load and transform image
        img = Image.open(image_path)
        transformed_image = weights(img).unsqueeze(0)
        original_image = transformed_image.clone()

        # Calculate ensemble predictions
        ensemble_outputs = []
        transformed_image.requires_grad = True
        for model in models:
            output = model(normalize(transformed_image))
            ensemble_outputs.append(output)

        ensemble_outputs = torch.stack(ensemble_outputs)
        ensemble_mean_output = torch.mean(ensemble_outputs, dim=0)

        # Get original predictions
        original_output = ensemble_mean_output.clone()
        original_probs = torch.softmax(original_output, dim=1)
        
        # Calculate loss
        loss = nn.CrossEntropyLoss()

        if targeted == True:
            target = torch.tensor([t])
            cost = -loss(original_output, target)
        else:
            target = original_output.max(1)[1]
            cost = -loss(original_output, target)
        
        # Calculate grad
        for model in models:
            model.zero_grad()
    
       # loss.backward()
        grad = torch.autograd.grad(cost, transformed_image)[0]

        # Generate adversarial image
        perturbed_image, adv_pert = ifgsm_attack(transformed_image, epsilon, grad)

        # Get perturbed predictions
        perturbed_output = torch.mean(torch.stack([model(normalize(perturbed_image)) for model in models]), dim=0)
        perturbed_probs = torch.softmax(perturbed_output, dim=1)

        # Sort predictions
        original_top_classes = original_probs[0].topk(num_classes)
        perturbed_top_classes = perturbed_probs[0].topk(num_classes)

        original_dict = {classes[idx.item()]: prob.item() for idx, prob in zip(original_top_classes.indices, original_top_classes.values)}
        perturbed_dict = {classes[idx.item()]: prob.item() for idx, prob in zip(perturbed_top_classes.indices, perturbed_top_classes.values)}

        results.append({'original': original_dict, 'perturbed': perturbed_dict})

        # Save adversarial image
        if folder:
            perturbed_image = perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            perturbed_image = (perturbed_image * 255).astype('uint8')
            Image.fromarray(perturbed_image).save(os.path.join(output_dir, image_name))

        # Plot if graph is True
        if graph:
            print(f'Epsilon: {epsilon}')
            if targeted == True:
                print(f'Targeted Attack = {classes[t]}')
            else:
                print(f'Untargeted Attack. Maximize Original Output')

            plt.subplot(1, 3, 1)
            plt.imshow(original_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Original: {max(original_dict, key=original_dict.get)} - {np.round(max(original_dict.values()), 3)}",  fontsize=7)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            if folder:
                plt.imshow(perturbed_image)
            else:
                plt.imshow(perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Perturbed: {max(perturbed_dict, key=perturbed_dict.get)} - {np.round(max(perturbed_dict.values()), 3)}", fontsize=7)
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(adv_pert.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Perturbation", fontsize=7)
            plt.axis('off') 

            plt.subplots_adjust(wspace=0.5)  # Add space between the plots
            plt.show()

    # Save results to txt
    if folder:
        with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
            for result in results:
                f.write(f"Original: {result['original']}\nPerturbed: {result['perturbed']}\n\n")

    return results


#3)-----------------ENSAMBLE ATTACK iFGSM + COURSE CLASSES-------------------------


def ensamble_attack_course_classes(image_folder, models, weights, epsilon, classes, targeted = False, t = 0, num_classes=3, graph=False, folder=False, control=False):
    """Test function to generate adversarial images and obtain predictions using an ensemble of models."""

    fine_to_coarse_mapping, unique_coarse = get_fine2course_mapping()

    for model in models:
        model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Prepare output directory
    if folder:
        if targeted == False:
            output_dir = f"ensamble-attack_ensamble-c_epsilon-{epsilon}-untargeted"
        else:
            output_dir = f"ensamble-attack_ensamble-c_epsilon-{epsilon}-targeted"
            
        os.makedirs(output_dir, exist_ok=True)

    results = []
    print('Iterating over images in folder')
    for image_name in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)

        # Load and transform image
        img = Image.open(image_path)
        transformed_image = weights(img).unsqueeze(0)
        original_image = transformed_image.clone()

        # Calculate ensemble predictions
        ensemble_outputs = []
        transformed_image.requires_grad = True
        for model in models:
            output = model(normalize(transformed_image))
            ensemble_outputs.append(output)

        ensemble_outputs = torch.stack(ensemble_outputs)
        ensemble_mean_output = torch.mean(ensemble_outputs, dim=0)

        # Get original predictions
        original_output = ensemble_mean_output.clone()
        original_probs = torch.softmax(original_output, dim=1)

        # Calculate loss
        loss = nn.CrossEntropyLoss()

        if targeted == True:
            target = torch.tensor([t])
            cost = -loss(original_output, target)
        else:
            target = original_output.max(1)[1]
            cost = -loss(original_output, target)

        #Calculate grad
        for model in models:
            model.zero_grad()

        #loss.backward()
        grad = torch.autograd.grad(cost, transformed_image)[0]

        # Generate adversarial image
        perturbed_image, adv_pert = ifgsm_attack(transformed_image, epsilon, grad)

        # Get perturbed predictions
        perturbed_output = torch.mean(torch.stack([model(normalize(perturbed_image)) for model in models]), dim=0)
        perturbed_probs = torch.softmax(perturbed_output, dim=1)

        # Sort predictions
        original_top_classes = original_probs[0].topk(num_classes)
        perturbed_top_classes = perturbed_probs[0].topk(num_classes)

        original_dict = {classes[idx.item()]: prob.item() for idx, prob in zip(original_top_classes.indices, original_top_classes.values)}
        perturbed_dict = {classes[idx.item()]: prob.item() for idx, prob in zip(perturbed_top_classes.indices, perturbed_top_classes.values)}

        # Compute coarse category scores
        coarse_scores = {}
        pert_coarse_scores = {}

        print('Fine Class --> Coarse class')
        for fine_class, coarse_class in tqdm(fine_to_coarse_mapping.items()):

            fine_indices = [classes.index(fine_class)]

            #original
            fine_logits = original_output[:, fine_indices]
            fine_logits = fine_logits.view(fine_logits.size(0), -1)
            other_indices = [idx for idx in range(original_output.size(1)) if idx not in fine_indices]
            other_logits = original_output[:, other_indices]
            coarse_scores[coarse_class] = (torch.logsumexp(fine_logits, dim = 1) - torch.logsumexp(other_logits, dim = 1))

            #perturbed
            fine_logits = perturbed_output[:, fine_indices]
            fine_logits = fine_logits.view(fine_logits.size(0), -1)
            other_indices = [idx for idx in range(perturbed_output.size(1)) if idx not in fine_indices]
            other_logits = perturbed_output[:, other_indices]
            pert_coarse_scores[coarse_class] = (torch.logsumexp(fine_logits, dim = 1) - torch.logsumexp(other_logits, dim = 1))

        #turn logits to probs
        b1 = torch.cat(tuple(coarse_scores.values()))
        coarse_scores = F.softmax(b1, dim = 0)

        b2 = torch.cat(tuple(pert_coarse_scores.values()))
        pert_coarse_scores = F.softmax(b2, dim = 0)

        #sorting values 
        sorted_values1, sorted_indices = torch.sort(coarse_scores, descending=True)
        sorted_keys1 = [unique_coarse[i] for i in sorted_indices]  

        sorted_values2, sorted_indices = torch.sort(pert_coarse_scores, descending=True)
        sorted_keys2 = [unique_coarse[i] for i in sorted_indices]        

        #define dicts with probs
        probs_scores = {}
        for key, value in zip(sorted_keys1, sorted_values1):
            probs_scores[key] = value.item() 

        probs_p_scores = {}
        for key, value in zip(sorted_keys2, sorted_values2):
            probs_p_scores[key] = value.item() 
        

        results.append({'original': original_dict, 'perturbed': perturbed_dict, 'coarse_scores': probs_scores, 'pert_coarse_scores': probs_p_scores})

        # Save adversarial image
        if folder:
            perturbed_image = perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            perturbed_image = (perturbed_image * 255).astype('uint8')
            Image.fromarray(perturbed_image).save(os.path.join(output_dir, image_name))

        # Plot if graph is True
        if graph:
            print(f'Epsilon: {epsilon}')
            if targeted == True:
                print(f'Targeted Attack = {classes[t]}')
            else:
                print(f'Untargeted Attack. Maximize Original Output')

            plt.subplot(1, 3, 1)
            plt.imshow(original_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Original: {max(probs_scores, key=probs_scores.get)} - {np.round(max(probs_scores.values()), 3)}",  fontsize=7)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            if folder:
                plt.imshow(perturbed_image)
            else:
                plt.imshow(perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Perturbed: {max(probs_p_scores, key=probs_p_scores.get)} - {np.round(max(probs_p_scores.values()), 3)}", fontsize=7)
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(adv_pert.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Perturbation", fontsize=7)
            plt.axis('off') 

            plt.subplots_adjust(wspace=0.5)  # Add space between the plots
            plt.show()

    # Save results to txt
    if folder:
        with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
            for result in results:
                f.write(f"Original: {result['original']}\nPerturbed: {result['perturbed']}\nCoarse Scores: {result['coarse_scores']}\nPerturbated Coarse Scores: {result['pert_coarse_scores']}\n")

    return results


