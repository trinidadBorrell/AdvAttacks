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
import json

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
        print(f"Classes file not found, try modifying the path in utils.py")
        return None

def get_default_weights():
    weights = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    return weights

def get_models_ensamble( models ):
    weights = [models.EfficientNet_B4_Weights.IMAGENET1K_V1, models.EfficientNet_B5_Weights.IMAGENET1K_V1, models.ResNet101_Weights.IMAGENET1K_V1, models.ResNet152_Weights.IMAGENET1K_V1, models.Inception_V3_Weights.IMAGENET1K_V1, models.ResNet50_Weights.IMAGENET1K_V1]
    models = [models.efficientnet_b4(weights=weights[0]), models.efficientnet_b5(weights = weights[1]), models.resnet101(weights=weights[2]), models.resnet152(weights=weights[3]), models.inception_v3(weights=weights[4]), models.resnet50(weights = weights[5])]
    return models

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
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights = weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights = weights)
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
def plot_results(folder_path, results, adv_attack = False, coarse_classes = False, number = 100):
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
        
def denorm(batch, mean, std):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean)
    if isinstance(std, list):
        std = torch.tensor(std)
    
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def ifgsm_attack(input, epsilon, data_grad):
    iter = int(min([epsilon + 4, epsilon * 1.25]))  # Number of iterations
    
    alpha = 1
    pert_out = input.clone().detach()
    
    for i in range(iter):
        pert_out = pert_out + (alpha / 255) * data_grad.sign()

        if torch.norm((pert_out - input), p=float('inf')) > epsilon / 255:
            break

    adv_pert = torch.clamp(input - pert_out, 0, 1)
    pert_out = torch.clamp(pert_out, 0, 1)

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

def simple_attack(image_folder, model, epsilon, classes, targeted = False, t = '0', num_classes=100, graph=False, folder=False, attack='iFGSM'):
    """Test function to generate adversarial images and obtain predictions."""
    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    auto_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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
        transformed_image = auto_transforms(img).unsqueeze(0)
        original_image = transformed_image.clone()
        original_image_denormed = denorm(original_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Calculate loss
        transformed_image.requires_grad = True
        output = model(normalize(transformed_image))
        probs = torch.softmax(output, dim=1)
        loss = nn.CrossEntropyLoss()

        if targeted == True:
            if t == '0':
                target = torch.argmax(probs, dim=1)    #maximize original output 
            else:
                target = torch.tensor([t])           #maximize target t
            cost = -loss(output, target)    
        else:
            true_class = output.max(1)[1]
            cost = loss(output, true_class)  #minimize original output

        model.zero_grad()
        grad = torch.autograd.grad(cost, transformed_image)[0]

        # Generate adversarial image
        transformed_image_denorm = denorm(transformed_image,  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        if attack == 'iFGSM':
            perturbed_image, adv_pert = ifgsm_attack(transformed_image_denorm, epsilon, grad)
        elif attack == 'FGSM':
            perturbed_image, adv_pert = fgsm_attack(transformed_image_denorm, epsilon, grad)

        # Get predictions
        original_output = model(original_image)
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
                if t == '0':
                    print(f'Targeted Attack. Maximize Original Output')
                else:
                    print(f'Targeted Attack = {classes[t]}')
            else:
                print(f'Untargeted Attack. Minimize Original Output')

            plt.subplot(1, 3, 1)
            plt.imshow(original_image_denormed.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
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
        with open(os.path.join(output_dir, 'results.json'), 'w') as json_file:
            json.dump(results, json_file)

    return results


#2)-----------------ENSAMBLE ATTACK iFGSM-------------------------
def ensamble_attack(image_folder, models, weights, epsilon, classes, targeted = False, t = '0', num_classes=100, graph=False, folder=False, attack = 'iFGSM'):
    """Test function to generate adversarial images and obtain predictions using an ensemble of models."""
    for model in models:
        model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    auto_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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
        transformed_image = auto_transforms(img).unsqueeze(0)
        original_image = transformed_image.clone()
        original_image_denormed = denorm(original_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #----------------

        # Calculate ensemble predictions
        ensemble_outputs = []
        transformed_image.requires_grad = True
        for model in models:
            output = model(transformed_image)
            ensemble_outputs.append(output)

        ensemble_outputs = torch.stack(ensemble_outputs)
        ensemble_mean_output = torch.mean(ensemble_outputs, dim=0)

        # Get original predictions
        original_output = ensemble_mean_output.clone()
        original_probs = torch.softmax(original_output, dim=1)
        
        # Calculate loss
        loss = nn.CrossEntropyLoss()

        if targeted == True:
            if t == '0':
                #target = original_output.max(1)[1]
                target = torch.argmax(original_probs, dim=1)   #maximize original output 
            else:
                target = torch.tensor([t])           #maximize target t
            cost = -loss(original_output, target)    
        else:
            true_class = original_output.max(1)[1]
            cost = loss(original_output, true_class)  #minimize original output

        # Calculate grad
        for model in models:
            model.zero_grad()
    
       # loss.backward()
        grad = torch.autograd.grad(cost, transformed_image)[0]

        # Generate adversarial image
        transformed_image_denorm = denorm(transformed_image,  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #-------------
        if attack == 'iFGSM':
            perturbed_image, adv_pert = ifgsm_attack(transformed_image_denorm, epsilon, grad)
        elif attack == 'FGSM':
            perturbed_image, adv_pert = fgsm_attack(transformed_image_denorm, epsilon, grad)

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
                if t == '0':
                    print(f'Targeted Attack. Maximize Original Output')
                else:
                    print(f'Targeted Attack = {classes[t]}')
            else:
                print(f'Untargeted Attack. Minimize Original Output')

            plt.subplot(1, 2, 1)
            plt.imshow(original_image_denormed.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Original: {max(original_dict, key=original_dict.get)} - {np.round(max(original_dict.values()), 3)}",  fontsize=7)
            plt.axis('off')

            plt.subplot(1, 2, 2)
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
        with open(os.path.join(output_dir, 'results.json'), 'w') as json_file:
            json.dump(results, json_file)

    return results


#3)-----------------ENSAMBLE ATTACK iFGSM + COaRSE CLASSES-------------------------

def get_coarse_arrays():
    labels = ['cat', 'dog', 'bird', 'bottle', 'chair', 'elephant', 'clock', 'truck', 'snake', 'spider', 'sheep']

    #cat : 6 (282 - 287) -> 6 
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    cat = np.array(list(range(281, 287)))

    #dog: 120 (152 - 269) -> 118 + (539) + dudoso: 277 African hunting dog
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    dog = np.array(list(range(151, 269)) + [537])

    #bird: 52 (8 - 25) -> 18 + (82 - 101) -> 20 + (129 - 148) -> 20
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    bird = np.array(list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147))) 

    #bottle: 7 (441) + (721) + (738) + (900) + (901) + (909) + (907)
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    bottle = np.array([440, 720, 737, 898, 899, 901, 907])

    #chair: 4 (424) + (560) + (766) 
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    chair = np.array([423, 559, 765, 857])

    #elephant: 2 (386 - 387) -> 2
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    elephant = np.array([385, 386])

    #clock: 3 (531) + (410) + (893) 
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    clock = np.array([409, 530, 892])

    #truck: 8 (556) + (570) + (718) + (865) + (868) + (676) + (657)
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    truck = np.array([555, 569, 656, 675, 717, 734, 864, 867])
    
    #snake: del 52 al 68 inclusive
    snake = np.arange(52, 69)

    #spider: del 72 al 77 inclusive
    spider = np.arange(72, 78)

    #sheep: del 348 al 352
    sheep = np.arange(348, 353)

    #index = [np.arange(8, 25), np.arange(151, 276), np.arange(280, 286)]
    index = [cat, dog, bird, bottle, chair, elephant, clock, truck, snake, spider, sheep]
    return labels, index

def eliminate_elements_torch(tensor, indices):
    # Use torch.masked_select() to select elements not in the specified indices
    mask = torch.ones_like(tensor, dtype=torch.bool)
    mask[indices] = 0
    result = torch.masked_select(tensor, mask)
    return result

def ensamble_attack_coarse_classes_sum(image_folder, models, weights, epsilon, classes, targeted = False, t = '0', num_classes=1000, graph=False, folder=False, sorted = True,  attack = 'iFGSM'):
    """Test function to generate adversarial images and obtain predictions using an ensemble of models."""

    c_classes, c_index = get_coarse_arrays() 

    for model in models:
        model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    auto_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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
        transformed_image = auto_transforms(img).unsqueeze(0)
        original_image = transformed_image.clone()
        original_image_denormed = denorm(original_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #----------------

        # Calculate ensemble predictions
        ensemble_outputs = []
        transformed_image.requires_grad = True
        for model in models:
            output = model(transformed_image)
            ensemble_outputs.append(output)

        ensemble_outputs = torch.stack(ensemble_outputs)
        ensemble_mean_output = torch.mean(ensemble_outputs, dim=0)

        # Get original predictions
        original_output = ensemble_mean_output.clone()
        original_probs = torch.softmax(original_output, dim=1)

        # Calculate loss
        loss = nn.CrossEntropyLoss()

        if targeted == True:
            if t == '0':
                target =torch.argmax(original_probs, dim=1)   #maximize original output 
            else:
                target = torch.tensor([t])           #maximize target t
            cost = -loss(original_output, target)    
        else:
            true_class = original_output.max(1)[1]
            cost = loss(original_output, true_class)  #minimize original output

        #Calculate grad
        for model in models:
            model.zero_grad()

        #loss.backward()
        grad = torch.autograd.grad(cost, transformed_image)[0]

        # Generate adversarial image
        transformed_image_denorm = denorm(transformed_image,  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        if attack == 'iFGSM':
            perturbed_image, adv_pert = ifgsm_attack(transformed_image_denorm, epsilon, grad)
        elif attack == 'FGSM':
            perturbed_image, adv_pert = fgsm_attack(transformed_image_denorm, epsilon, grad)

        # Get perturbed predictions
        perturbed_output = torch.mean(torch.stack([model(normalize(perturbed_image)) for model in models]), dim=0)
        perturbed_probs = torch.softmax(perturbed_output, dim=1)

        # Sort predictions
        original_top_classes = original_probs[0].topk(num_classes)
        perturbed_top_classes = perturbed_probs[0].topk(num_classes)

        original_dict = {classes[idx.item()]: prob.item() for idx, prob in zip(original_top_classes.indices, original_top_classes.values)}
        perturbed_dict = {classes[idx.item()]: prob.item() for idx, prob in zip(perturbed_top_classes.indices, perturbed_top_classes.values)}

        # Compute coarse category scores
        print('Fine Class --> Coarse class')

        scores_original = torch.zeros(size= (len(c_index),))
        scores_perturbed = torch.zeros(size= (len(c_index),))

        for i in tqdm(np.arange(len(c_index))): #Iterate in the coarse classes

            c = c_index[i] #coarse class indexs

            c_values = original_probs[:, c]
            sum = torch.sum(c_values)
            scores_original[i] = sum

            c_values_p = perturbed_probs[:, c]
            sum_p = torch.sum(c_values_p)
            scores_perturbed[i] = sum_p


        #Eliminate values of probs (perturbed and original)
        coarse_idx = np.hstack(c_index) #index to eliminate
        coarse_scores = eliminate_elements_torch(original_probs[0], coarse_idx)
        pert_coarse_scores = eliminate_elements_torch(perturbed_probs[0], coarse_idx)

        new_classes = [string for idx, string in enumerate(classes) if idx not in coarse_idx]

        #Concat new scores and new list of classes
        coarse_scores = torch.cat((coarse_scores, scores_original), dim = 0)
        pert_coarse_scores = torch.cat((pert_coarse_scores,  scores_perturbed), dim = 0)

        new_classes = new_classes + c_classes

        #sorting values 
        sorted_probs, sorted_indices = torch.sort(coarse_scores, descending=True)
        sorted_classes = [new_classes[i] for i in sorted_indices]

        sorted_probs_p, sorted_indices_p = torch.sort(pert_coarse_scores, descending=True)
        sorted_classes_p = [new_classes[i] for i in sorted_indices_p]

        #define dicts with probs
        probs_scores = {}
        for key, value in zip(sorted_classes, sorted_probs):
            probs_scores[key] = value.item() 

        probs_p_scores = {}
        for key, value in zip(sorted_classes_p, sorted_probs_p):
            probs_p_scores[key] = value.item() 

        results.append({'original': original_dict, 'perturbed': perturbed_dict, 'coarse_scores': probs_scores, 'pert_coarse_scores': probs_p_scores})

        # Save adversarial image
        if folder:
            perturbed_image = perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            perturbed_image = (perturbed_image * 255).astype('uint8')
            Image.fromarray(perturbed_image).save(os.path.join(output_dir, image_name))
    
        if targeted == True:
            if t == '0':
                print(f'Targeted Attack. Maximize Original Output')
            else:
                print(f'Targeted Attack = {classes[t]}\nOriginal prob target: {probs_scores[classes[t]]},\nPerturbed prob target: {probs_p_scores[classes[t]]}')
        else:
            print(f'Untargeted Attack. Minimize Original Output')

        # Plot if graph is True
        if graph:
            print(f'Epsilon: {epsilon}')

            plt.subplot(1, 3, 1)
            plt.imshow(original_image_denormed.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Original: {max(original_dict, key=original_dict.get)} - {np.round(max(original_dict.values()), 3)}" + "\n" + f"Original: {max(probs_scores, key=probs_scores.get)} - {np.round(max(probs_scores.values()), 3)}", fontsize=7)
            plt.axis('off')
         #   print(f'C-Scores: {coarse_scores[sorted_indices[:5]]} - {sorted_classes[:5]}')
            plt.subplot(1, 3, 2)
            if folder:
                plt.imshow(perturbed_image)
            else:
                plt.imshow(perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Perturbed: {max(perturbed_dict, key=perturbed_dict.get)} - {np.round(max(perturbed_dict.values()), 3)}" + "\n" + f"Perturbed: {max(probs_p_scores, key=probs_p_scores.get)} - {np.round(max(probs_p_scores.values()), 3)}", fontsize=7)
          #  print(f'Perturbed C-Scores: {pert_coarse_scores[sorted_indices_p[:5]]} - {sorted_classes_p[:5]}')

            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(adv_pert.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Perturbation", fontsize=7)
            plt.axis('off') 

            plt.subplots_adjust(wspace=0.5)  # Add space between the plots
            plt.show()

    # Save results to txt
    if folder:
        with open(os.path.join(output_dir, 'results.json'), 'w') as json_file:
            json.dump(results, json_file)

    return results

#Generate the weights when considering more than one target

def generate_list(length, specified_positions, weights):
    weights = weights[0]
    output_list = [0] * length
    for i, pos in enumerate(specified_positions):
        output_list[pos] = weights[i]
    return output_list

def extract_and_normalize(original_probs, positions):
    # Extract values at specified positions
    extracted_values = original_probs[:, positions]
    # Sum of extracted values
    sum_values = torch.sum(extracted_values)
    # Normalize extracted values
    weights = extracted_values / sum_values
    return weights.tolist()

def extract_values(original_list, specified_indexes):
    # Use list comprehension to filter out values not corresponding to specified indexes
    extracted_values = [original_list[i] for i in range(len(original_list)) if i not in specified_indexes]
    return extracted_values

#Phase randomization for control image

#Complete coarse attack

def ensamble_attack_coarse_classes_sum_full(image_folder, models, weights, epsilon, classes, targeted = False, t = '0', coarse_class = 'nada', num_classes=1000, graph=False, folder=False, sorted = True,  attack = 'iFGSM'):
    """Test function to generate adversarial images and obtain predictions using an ensemble of models."""

    c_classes, c_index = get_coarse_arrays() 


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
            model.eval()
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
            if type(t) == str:
                target = original_output.max(1)[1]   #maximize original output
                cost = -loss(original_output, target)
            elif type(t) == int:
                target = torch.tensor([t])           #maximize target t
                cost = -loss(original_output, target)
              #  print(f'Final cost: {cost}')    
            elif type(t) == np.ndarray or type(t) == list:
                w = extract_and_normalize(original_probs, t)
                probs_t = generate_list(1000, t, w)
                target = torch.tensor([probs_t])           
                cost = -loss(original_output, target)
        else:
            if type(t) != str:
                classes_pos = np.arange(0, 1000)
                non_t = extract_values(np.arange(1000), t)
                w = extract_and_normalize(original_probs, non_t)
                probs_t = generate_list(1000, non_t, w)
                target = torch.tensor([probs_t])           
                cost = -loss(original_output, target)
            else:
                target = original_output.max(1)[1]
                cost = loss(original_output, target)  #minimize original output

        #Calculate grad
        for model in models:
            model.zero_grad()

        #loss.backward()
        grad = torch.autograd.grad(cost, transformed_image)[0]

        # Generate adversarial image
        if attack == 'iFGSM':
            perturbed_image, adv_pert = ifgsm_attack(transformed_image, epsilon, grad)
        elif attack == 'FGSM':
            perturbed_image, adv_pert = fgsm_attack(transformed_image, epsilon, grad)

        # Get perturbed predictions
        perturbed_output = torch.mean(torch.stack([model(normalize(perturbed_image)) for model in models]), dim=0)
        perturbed_probs = torch.softmax(perturbed_output, dim=1)

        # Sort predictions
        original_top_classes = original_probs[0].topk(num_classes)
        perturbed_top_classes = perturbed_probs[0].topk(num_classes)

        original_dict = {classes[idx.item()]: prob.item() for idx, prob in zip(original_top_classes.indices, original_top_classes.values)}
        perturbed_dict = {classes[idx.item()]: prob.item() for idx, prob in zip(perturbed_top_classes.indices, perturbed_top_classes.values)}

        # Compute coarse category scores
        print('Fine Class --> Coarse class')

        scores_original = torch.zeros(size= (len(c_index),))
        scores_perturbed = torch.zeros(size= (len(c_index),))

        for i in tqdm(np.arange(len(c_index))): #Iterate in the coarse classes

            c = c_index[i] #coarse class indexs

            c_values = original_probs[:, c]
            sum = torch.sum(c_values)
            scores_original[i] = sum

            c_values_p = perturbed_probs[:, c]
            sum_p = torch.sum(c_values_p)
            scores_perturbed[i] = sum_p


        #Eliminate values of probs (perturbed and original)
        coarse_idx = np.hstack(c_index) #index to eliminate
      #  print(original_probs.shape, original_probs[0].shape)
        coarse_scores = eliminate_elements_torch(original_probs[0], coarse_idx)
        pert_coarse_scores = eliminate_elements_torch(perturbed_probs[0], coarse_idx)

        new_classes = [string for idx, string in enumerate(classes) if idx not in coarse_idx]

        #Concat new scores and new list of classes
        coarse_scores = torch.cat((coarse_scores, scores_original), dim = 0)
        pert_coarse_scores = torch.cat((pert_coarse_scores,  scores_perturbed), dim = 0)

        new_classes = new_classes + c_classes

        #sorting values 
        sorted_probs, sorted_indices = torch.sort(coarse_scores, descending=True)
        sorted_classes = [new_classes[i] for i in sorted_indices]

        sorted_probs_p, sorted_indices_p = torch.sort(pert_coarse_scores, descending=True)
        sorted_classes_p = [new_classes[i] for i in sorted_indices_p]

        #define dicts with probs
        probs_scores = {}
        for key, value in zip(sorted_classes, sorted_probs):
            probs_scores[key] = value.item() 

        probs_p_scores = {}
        for key, value in zip(sorted_classes_p, sorted_probs_p):
            probs_p_scores[key] = value.item() 

        results.append({'original': original_dict, 'perturbed': perturbed_dict, 'coarse_scores': probs_scores, 'pert_coarse_scores': probs_p_scores})

        # Save adversarial image
        if folder:
            perturbed_image = perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            perturbed_image = (perturbed_image * 255).astype('uint8')
            Image.fromarray(perturbed_image).save(os.path.join(output_dir, image_name))


        print(f'Epsilon: {epsilon}')
        if targeted == True:
            if type(t) == str:
                print(f'Targeted Attack. Maximize Original Output')
            elif len(t) == 1:
                print(f'Targeted Attack = {classes[t[0]]}\nOriginal prob target: {probs_scores[classes[t[0]]]},\nPerturbed prob target: {probs_p_scores[classes[t[0]]]}')
            elif len(t) > 1:
                print(f'Targeted Attack = {coarse_class}\nOriginal prob target: {probs_scores[coarse_class]},\nPerturbed prob target: {probs_p_scores[coarse_class]}')
        else:
            print(f'Untargeted Attack. Minimize Original Output\nOriginal prob target: {probs_scores[sorted_classes[0]]},\nPerturbed prob target: {probs_p_scores[sorted_classes[0]]}')

        # Plot if graph is True
        if graph:
            plt.subplot(1, 2, 1)
            plt.imshow(original_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Original: {max(original_dict, key=original_dict.get)} - {np.round(max(original_dict.values()), 3)}" + "\n" + f"Original: {max(probs_scores, key=probs_scores.get)} - {np.round(max(probs_scores.values()), 3)}", fontsize=7)
            plt.axis('off')
         #   print(f'C-Scores: {coarse_scores[sorted_indices[:5]]} - {sorted_classes[:5]}')
            plt.subplot(1, 2, 2)
            if folder:
                plt.imshow(perturbed_image)
            else:
                plt.imshow(perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Perturbed: {max(perturbed_dict, key=perturbed_dict.get)} - {np.round(max(perturbed_dict.values()), 3)}" + "\n" + f"Perturbed: {max(probs_p_scores, key=probs_p_scores.get)} - {np.round(max(probs_p_scores.values()), 3)}", fontsize=7)
          #  print(f'Perturbed C-Scores: {pert_coarse_scores[sorted_indices_p[:5]]} - {sorted_classes_p[:5]}')

            plt.axis('off')
            
          #  plt.subplot(1, 3, 3)
          #  plt.imshow(adv_pert.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
          #  plt.title(f"Perturbation", fontsize=7)
          #  plt.axis('off') 

            plt.subplots_adjust(wspace=0.5)  # Add space between the plots
            plt.show()

    # Save results to txt
    if folder:
        with open(os.path.join(output_dir, 'results.json'), 'w') as json_file:
            json.dump(results, json_file)

    return results


def randomize_phase_torch(image_tensor):
    # Convert torch tensor to NumPy array
    image_np = image_tensor.detach().numpy()

    # Separate color channels
    red_channel = image_np[:, 0, :, :]
    green_channel = image_np[:, 1, :, :]
    blue_channel = image_np[:, 2, :, :]

    # Compute FFT for each color channel
    red_fft = np.fft.fft2(red_channel, axes=(-2, -1))
    green_fft = np.fft.fft2(green_channel, axes=(-2, -1))
    blue_fft = np.fft.fft2(blue_channel, axes=(-2, -1))

    # Randomize phases for each color channel
    red_phases = np.angle(red_fft)
    green_phases = np.angle(green_fft)
    blue_phases = np.angle(blue_fft)

    randomized_red_phases = np.exp(1j * np.random.uniform(-np.pi, np.pi, red_phases.shape))
    randomized_green_phases = np.exp(1j * np.random.uniform(-np.pi, np.pi, green_phases.shape))
    randomized_blue_phases = np.exp(1j * np.random.uniform(-np.pi, np.pi, blue_phases.shape))

    # Combine magnitudes with randomized phases for each color channel
    randomized_red_fft = np.abs(red_fft) * randomized_red_phases
    randomized_green_fft = np.abs(green_fft) * randomized_green_phases
    randomized_blue_fft = np.abs(blue_fft) * randomized_blue_phases

    # Compute the inverse FFT for each color channel
    randomized_red_channel = np.fft.ifft2(randomized_red_fft, axes=(-2, -1))
    randomized_green_channel = np.fft.ifft2(randomized_green_fft, axes=(-2, -1))
    randomized_blue_channel = np.fft.ifft2(randomized_blue_fft, axes=(-2, -1))

    # Take the abs to get the image for each color channel
    randomized_red_channel = np.abs(randomized_red_channel)
    randomized_green_channel = np.abs(randomized_green_channel)
    randomized_blue_channel = np.abs(randomized_blue_channel)

    # Combine color channels into RGB tensor
    randomized_image = np.stack((randomized_red_channel, randomized_green_channel, randomized_blue_channel), axis=1)
    randomized_image_plot = torch.from_numpy(randomized_image)

    return randomized_image_plot

def ifgsm_attack_v2(input, epsilon, data_grad, control = True, mode ='flip'):
    iter = int(min([epsilon + 4, epsilon * 1.25]))  # Number of iterations
    
    alpha = 1
    pert_out = input.clone().detach()
    
    for i in range(iter):
        pert_out = pert_out + (alpha / 255) * data_grad.sign()

        if torch.norm((pert_out - input), p=float('inf')) > epsilon / 255:
            break

    adv_pert = (pert_out - input)
    adv_pert_max = torch.max(adv_pert)
    adv_pert_min = torch.min(adv_pert)
    adv_pert_plot = (adv_pert - adv_pert_min)/(adv_pert_max - adv_pert_min)
    normalization = torch.norm(pert_out - input)

    pert_out_plot = pert_out

    if control == True:
            if mode == 'flip':
                control_pert = torch.flip(adv_pert, [3])
                control_image = control_pert + input
                control_pert_plot = torch.flip(adv_pert_plot, [3])
                
            elif mode == 'rndm':
                control_pert =  randomize_phase_torch(adv_pert_plot)
                print(type(control_pert), np.shape(control_pert))
                control_image = control_pert + input
                control_image.double()

    return pert_out_plot, adv_pert_plot, control_image, control_pert_plot 

def ensamble_attack_coarse_classes_sum_full_v2(image_folder, models, weights, epsilon, classes, targeted = False, t = '0', coarse_class = 'nada', num_classes=1000, graph=False, folder=False, sorted = True,  attack = 'iFGSM'):
    """Test function to generate adversarial images and obtain predictions using an ensemble of models."""

    c_classes, c_index = get_coarse_arrays() 


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
    control = []
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
            model.eval()
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
            if type(t) == str:
                target = original_output.max(1)[1]   #maximize original output
                cost = -loss(original_output, target)
            elif type(t) == int:
                target = torch.tensor([t])           #maximize target t
                cost = -loss(original_output, target)

            elif type(t) == np.ndarray or type(t) == list:
                w = extract_and_normalize(original_probs, t)
                probs_t = generate_list(1000, t, w)
                target = torch.tensor([probs_t])           
                cost = -loss(original_output, target)
        else:
            if type(t) != str:
                non_t = extract_values(np.arange(1000), t)
                w = extract_and_normalize(original_probs, non_t)
                probs_t = generate_list(1000, non_t, w)
                target = torch.tensor([probs_t])           
                cost = -loss(original_output, target)
            else:
                target = original_output.max(1)[1]
                cost = loss(original_output, target)  #minimize original output

        #Calculate grad
        for model in models:
            model.zero_grad()

        #loss.backward()
        grad = torch.autograd.grad(cost, transformed_image)[0]

        # Generate adversarial image
        if attack == 'iFGSM':
            perturbed_image, adv_pert, control_image, control_pert = ifgsm_attack_v2(transformed_image, epsilon, grad)
        elif attack == 'FGSM':
            perturbed_image, adv_pert, control_image, control_pert = fgsm_attack(transformed_image, epsilon, grad)

        # Get perturbed predictions
        perturbed_output = torch.mean(torch.stack([model(normalize(perturbed_image)) for model in models]), dim=0)
        perturbed_probs = torch.softmax(perturbed_output, dim=1)

        control_output = torch.mean(torch.stack([model(normalize(control_image)) for model in models]), dim=0)
        control_probs = torch.softmax(control_output, dim=1)

        # Sort predictions
        original_top_classes = original_probs[0].topk(num_classes)
        perturbed_top_classes = perturbed_probs[0].topk(num_classes)
        control_top_classes = control_probs[0].topk(num_classes)

        original_dict = {classes[idx.item()]: prob.item() for idx, prob in zip(original_top_classes.indices, original_top_classes.values)}
        perturbed_dict = {classes[idx.item()]: prob.item() for idx, prob in zip(perturbed_top_classes.indices, perturbed_top_classes.values)}
        control_dict = {classes[idx.item()]: prob.item() for idx, prob in zip(control_top_classes.indices, control_top_classes.values)}

        # Compute coarse category scores
        print('Fine Class --> Coarse class')

        scores_original = torch.zeros(size= (len(c_index),))
        scores_perturbed = torch.zeros(size= (len(c_index),))
        scores_control = torch.zeros(size= (len(c_index),))

        for i in tqdm(np.arange(len(c_index))): #Iterate in the coarse classes

            c = c_index[i] #coarse class indexs

            c_values = original_probs[:, c]
            sum = torch.sum(c_values)
            scores_original[i] = sum

            c_values_p = perturbed_probs[:, c]
            sum_p = torch.sum(c_values_p)
            scores_perturbed[i] = sum_p

            c_values_c = control_probs[:, c]
            sum_c = torch.sum(c_values_c)
            scores_control[i] = sum_c

        #Eliminate values of probs (perturbed and original)
        coarse_idx = np.hstack(c_index) #index to eliminate
        coarse_scores = eliminate_elements_torch(original_probs[0], coarse_idx)
        pert_coarse_scores = eliminate_elements_torch(perturbed_probs[0], coarse_idx)
        ctrl_coarse_scores = eliminate_elements_torch(control_probs[0], coarse_idx)

        new_classes = [string for idx, string in enumerate(classes) if idx not in coarse_idx]

        #Concat new scores and new list of classes
        coarse_scores = torch.cat((coarse_scores, scores_original), dim = 0)
        pert_coarse_scores = torch.cat((pert_coarse_scores,  scores_perturbed), dim = 0)
        ctrl_coarse_scores = torch.cat((ctrl_coarse_scores,  scores_control), dim = 0)

        new_classes = new_classes + c_classes

        #sorting values 
        sorted_probs, sorted_indices = torch.sort(coarse_scores, descending=True)
        sorted_classes = [new_classes[i] for i in sorted_indices]

        sorted_probs_p, sorted_indices_p = torch.sort(pert_coarse_scores, descending=True)
        sorted_classes_p = [new_classes[i] for i in sorted_indices_p]
        
        sorted_probs_c, sorted_indices_c = torch.sort(ctrl_coarse_scores, descending=True)
        sorted_classes_c = [new_classes[i] for i in sorted_indices_c]

        #define dicts with probs
        probs_scores = {}
        for key, value in zip(sorted_classes, sorted_probs):
            probs_scores[key] = value.item() 

        probs_p_scores = {}
        for key, value in zip(sorted_classes_p, sorted_probs_p):
            probs_p_scores[key] = value.item() 

        probs_c_scores = {}
        for key, value in zip(sorted_classes_c, sorted_probs_c):
            probs_c_scores[key] = value.item() 

        results.append({'original': original_dict, 'perturbed': perturbed_dict, 'coarse_scores': probs_scores, 'pert_coarse_scores': probs_p_scores})
        control.append({'ctrl': control_dict, 'ctrl_scores': probs_c_scores})

        #print(probs_c_scores)

        # Save adversarial image
        if folder:
            counter = 1
            perturbed_image = perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            perturbed_image = (perturbed_image * 255).astype('uint8')

            adv_pert = adv_pert.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            adv_pert = (adv_pert * 255).astype('uint8')

            control_image = control_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            control_image = (control_image * 255).astype('uint8')

            control_pert = control_pert.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            control_pert = (control_pert * 255).astype('uint8')

            while True:
                # Generate folder name
                folder_name = f"images_folder_{counter}"
                folder_path = os.path.join(output_dir, folder_name)

                # Check if folder already exists
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    break
                else:
                    counter += 1

            atk_path = os.path.join(folder_path, 'atk.png')
            Image.fromarray(perturbed_image).save(atk_path)

            pert_path = os.path.join(folder_path, 'pert.png')
            Image.fromarray(adv_pert).save(pert_path)

            control_pert_path = os.path.join(folder_path, 'control_pert.png')
            Image.fromarray(control_pert).save(control_pert_path)

            control_path = os.path.join(folder_path, 'control.png')
            Image.fromarray(control_image).save(control_path)


        print(f'Epsilon: {epsilon}')
        if targeted == True:
            if type(t) == str:
                print(f'Targeted Attack. Maximize Original Output\nOriginal prob: {probs_scores[sorted_classes[0]]},\nPerturbed prob: {probs_p_scores[sorted_classes[0]]}\nControl prob: {probs_c_scores[sorted_classes[0]]}')
            elif len(t) == 1:
                print(f'Targeted Attack = {classes[t[0]]}\nOriginal prob target: {probs_scores[classes[t[0]]]},\nPerturbed prob target: {probs_p_scores[classes[t[0]]]}\nControl prob target: {probs_c_scores[classes[t[0]]]}')
            elif len(t) > 1:
                print(f'Targeted Attack = {coarse_class}\nOriginal prob target: {probs_scores[coarse_class]},\nPerturbed prob target: {probs_p_scores[coarse_class]}\nControl prob target: {probs_c_scores[coarse_class]}')
        else:
            print(f'Untargeted Attack. Minimize Original Output\nOriginal prob target: {probs_scores[sorted_classes[0]]},\nPerturbed prob target: {probs_p_scores[sorted_classes[0]]}\nControl prob target: {probs_c_scores[sorted_classes[0]]}')

        # Plot if graph is True
        if graph:

            if folder != True:
                perturbed_image = perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                adv_pert_plot = adv_pert.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                control_image = control_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                control_pert = control_pert.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

            plt.subplot(1, 3, 1)
            plt.imshow(original_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Original: {max(original_dict, key=original_dict.get)} - {np.round(max(original_dict.values()), 3)}" + "\n" + f"Original: {max(probs_scores, key=probs_scores.get)} - {np.round(max(probs_scores.values()), 3)}", fontsize=7)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(np.clip(perturbed_image, 0, 1))
            plt.title(f"Perturbed: {max(perturbed_dict, key=perturbed_dict.get)} - {np.round(max(perturbed_dict.values()), 3)}" + "\n" + f"Perturbed: {max(probs_p_scores, key=probs_p_scores.get)} - {np.round(max(probs_p_scores.values()), 3)}", fontsize=7)
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(np.clip(control_image, 0, 1))
            plt.title(f"Control Image: {max(control_dict, key=control_dict.get)} - {np.round(max(control_dict.values()), 3)}" + "\n" + f"Control: {max(probs_c_scores, key=probs_c_scores.get)} - {np.round(max(probs_c_scores.values()), 3)}", fontsize=7)
            plt.axis('off') 

            plt.subplots_adjust(wspace=0.5)  # Add space between the plots
            plt.show()

            plt.subplot(1, 2, 1)
            plt.imshow(adv_pert_plot)
            plt.title(f"Perturbation", fontsize=7)
            plt.axis('off') 

            plt.subplot(1, 2, 2)
            plt.imshow(control_pert)
            plt.title(f"Perturbation - Control", fontsize=7)
            plt.axis('off') 

            plt.subplots_adjust(wspace=0.5)  # Add space between the plots
            plt.show()

    # Save results to txt
    if folder:
        with open(os.path.join(folder_path, 'results.json'), 'w') as json_file:
            json.dump(results, json_file)

    return results

#OPENING RESULTS WHEN SAVED

# with open(os.path.join(res_path, 'results.json'), 'r') as json_file:
#    loaded_results = json.load(json_file)