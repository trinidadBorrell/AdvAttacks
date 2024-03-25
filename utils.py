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

def get_models_ensamble( models ):
    weights = [models.EfficientNet_B4_Weights.IMAGENET1K_V1, models.EfficientNet_B5_Weights.IMAGENET1K_V1, models.ResNet101_Weights.IMAGENET1K_V1, models.ResNet152_Weights.IMAGENET1K_V1, models.Inception_V3_Weights.IMAGENET1K_V1]
    models = [models.efficientnet_b4(weights=weights[0]), models.efficientnet_b5(weights = weights[1]), models.resnet101(weights=weights[2]), models.resnet152(weights=weights[3]), models.inception_v3(weights=weights[4])]
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

def simple_attack(image_folder, model, weights, epsilon, classes, targeted = False, t = 0, num_classes=100, graph=False, folder=False, control=False):
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
def ensamble_attack(image_folder, models, weights, epsilon, classes, targeted = False, t = 0, num_classes=100, graph=False, folder=False, control=False):
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

def get_course_arrays():
    labels = ['cat', 'dog', 'bird', 'bottle', 'sheep', 'chair', 'elephant', 'clock', 'truck']

    #cat : 6 (282 - 286) -> 5 + dudoso: 385 
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    cat = np.array(list(range(281, 286)) + [383])

    #dog: 120 (152 - 269) -> 118 + (539) + dudoso: 277 African hunting dog
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    dog = np.array(list(range(151, 269)) + [275, 537])

    #bird: 52 (8 - 25) -> 18 + (82 - 101) -> 20 + (129 - 148) -> 20
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    bird = np.array(list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147))) 

    #bottle: 7 (441) + (721) + (738) + (900) + (909) + dudosas : (456: bottlecap y 513: bottlescrew)
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    bottle = np.array([440, 455, 512, 720, 737, 898, 907])

    #sheep: 6 (350 -351) -> 2 
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    sheep = np.array([348, 349])

    #chair: 4 (424) + (560) + (766) 
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    chair = np.array([423, 559, 765])

    #elephant: 2 (386 - 387) -> 2
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    elephant = np.array([385, 386])

    #clock: 3 (531) + (410) + (893) 
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    clock = np.array([409, 530, 892])

    #truck: 8 (556) + (570) + (718) + (865) + (868) + (676) + (657)
    #obs: cambia con respecto a los valores de arriba porque las lineas en imageclassnet =/= a la ubicación en la lista importada
    truck = np.array([555, 569, 656, 675, 717, 864, 867])

    #index = [np.arange(8, 25), np.arange(151, 276), np.arange(280, 286)]
    index = [cat, dog, bird, bottle, sheep, chair, elephant, clock, truck]
    return labels, index

def eliminate_elements_torch(tensor, indices):
    # Use torch.masked_select() to select elements not in the specified indices
    mask = torch.ones_like(tensor, dtype=torch.bool)
    mask[indices] = 0
    result = torch.masked_select(tensor, mask)
    return result

def ensamble_attack_course_classes(image_folder, models, weights, epsilon, classes, targeted = False, t = 0, num_classes=100, graph=False, folder=False, control=False):
    """Test function to generate adversarial images and obtain predictions using an ensemble of models."""

    c_classes, c_index = get_course_arrays() 

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
        print('Fine Class --> Coarse class')

        scores_original = torch.zeros(size= (len(c_index),))
        scores_perturbed = torch.zeros(size= (len(c_index),))
        all_scores = torch.zeros(size= (len(classes),))
        all_scores_p = torch.zeros(size= (len(classes),))

        for i in np.arange(1000):
            c = np.array([i]) #course class indexs
            non_c = np.delete(np.arange(1000), i) #non course class index

            #Calculate Scores Course Clases - Non perturbed image
            c_logit = original_output[:, c]
            Si = torch.logsumexp(c_logit, dim = 1) 
            non_c_logit = original_output[:, non_c]
            Sj = torch.logsumexp(non_c_logit, dim = 1)
            all_scores[i] = Si - Sj

            #Calculate Scores Course Clases - Perturbed image
            c_logit_p = perturbed_output[:, c]
            Si_p = torch.logsumexp(c_logit_p, dim = 1) 
            non_c_logit_p = original_output[:, non_c]
            Sj_p = torch.logsumexp(non_c_logit_p, dim = 1)

           # scores_perturbed[i] = F.softmax(Si_p - Sj_p, dim = 0)
            # scores_perturbed[i] = Si_p - Sj_p / torch.logsumexp(original_output, dim = 1)
            all_scores_p[i] = Si_p - Sj_p


        for i in tqdm(np.arange(len(c_index))): #Iterate in the course classes

            c = c_index[i] #course class indexs
            non_c = np.delete(np.arange(1000), c) #non course class index

            #Calculate Scores Course Clases - Non perturbed image
            c_logit = original_output[:, c]

            Si = torch.logsumexp(c_logit, dim = 1) 

            non_c_logit = original_output[:, non_c]

            Sj = torch.logsumexp(non_c_logit, dim = 1)

            #scores_original[i] = F.softmax(Si - Sj, dim = 0)
            scores_original[i] = Si - Sj 
           # print(Si - Sj, Si - Sj / torch.logsumexp(original_output, dim = 1))
           # print(torch.logsumexp(original_output, dim = 1))
            #Calculate Scores Course Clases - Perturbed image
            c_logit_p = perturbed_output[:, c]
            Si_p = torch.logsumexp(c_logit_p, dim = 1) 

            non_c_logit_p = original_output[:, non_c]
            Sj_p = torch.logsumexp(non_c_logit_p, dim = 1)

           # scores_perturbed[i] = F.softmax(Si_p - Sj_p, dim = 0)
            # scores_perturbed[i] = Si_p - Sj_p / torch.logsumexp(original_output, dim = 1)
            scores_perturbed[i] = Si_p - Sj_p

        #Eliminate values of probs (perturbed and original)
        coarse_idx = np.hstack(c_index) #index to eliminate

        coarse_scores = eliminate_elements_torch(all_scores, coarse_idx)
        pert_coarse_scores = eliminate_elements_torch(all_scores_p, coarse_idx)

        new_classes = [string for idx, string in enumerate(classes) if idx not in coarse_idx]

        #Concat new scores and new list of classes
        #Turn results into probabilities
        coarse_scores = torch.cat((coarse_scores, scores_original), dim = 0)
        pert_coarse_scores = torch.cat((pert_coarse_scores,  scores_perturbed), dim = 0)

        coarse_scores = torch.softmax(coarse_scores, dim = 0)
        pert_coarse_scores = torch.softmax(pert_coarse_scores, dim = 0)

      #  coarse_scores = torch.cat((coarse_scores, torch.unsqueeze(scores_original, dim=0)), dim = 1)
      #  pert_coarse_scores = torch.cat((pert_coarse_scores,  torch.unsqueeze(scores_perturbed, dim = 0)), dim = 1)
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
      #  print(f'Fine dicts: {original_dict}, {perturbed_dict}')
      #  print(f'Coarse dicts: {probs_p_scores}, {probs_p_scores}')

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
            plt.title(f"Original: {max(original_dict, key=original_dict.get)} - {np.round(max(original_dict.values()), 3)}" + "\n" + f"Original: {max(probs_scores, key=probs_scores.get)} - {np.round(max(probs_scores.values()), 3)}", fontsize=7)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            if folder:
                plt.imshow(perturbed_image)
            else:
                plt.imshow(perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Perturbed: {max(perturbed_dict, key=perturbed_dict.get)} - {np.round(max(perturbed_dict.values()), 3)}" + "\n" + f"Perturbed: {max(probs_p_scores, key=probs_p_scores.get)} - {np.round(max(probs_p_scores.values()), 3)}", fontsize=7)
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

def ensamble_attack_course_classes_sum(image_folder, models, weights, epsilon, classes, targeted = False, t = 0, num_classes=100, graph=False, folder=False, control=False):
    """Test function to generate adversarial images and obtain predictions using an ensemble of models."""

    c_classes, c_index = get_course_arrays() 

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
        print('Fine Class --> Coarse class')

        scores_original = torch.zeros(size= (len(c_index),))
        scores_perturbed = torch.zeros(size= (len(c_index),))

        for i in tqdm(np.arange(len(c_index))): #Iterate in the course classes

            c = c_index[i] #course class indexs

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
        #print(coarse_scores.sum(), pert_coarse_scores.sum())
        # Plot if graph is True
        if graph:
            print(f'Epsilon: {epsilon}')
            if targeted == True:
                print(f'Targeted Attack = {classes[t]}')
            else:
                print(f'Untargeted Attack. Maximize Original Output')

            plt.subplot(1, 3, 1)
            plt.imshow(original_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
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
        with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
            for result in results:
                f.write(f"Original: {result['original']}\nPerturbed: {result['perturbed']}\nCoarse Scores: {result['coarse_scores']}\nPerturbated Coarse Scores: {result['pert_coarse_scores']}\n")

    return results


def plot_results_ensamble(results, n_classes):
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



