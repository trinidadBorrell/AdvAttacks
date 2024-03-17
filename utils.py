import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn.functional as F

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
def plot_results(folder_path, results):
    # Get list of image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]

    # Check if number of images matches number of results
    if len(image_files) != len(results):
        raise ValueError("Number of images in folder does not match number of results.")

    for image_file, res_dict in zip(image_files, results):
        image_path = os.path.join(folder_path, image_file)
        classes = list(res_dict.keys())
        probabilities = list(res_dict.values())

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
        pert_out = torch.clamp(pert_out, input - epsilon / 255, input + epsilon / 255).requires_grad_()

        if torch.norm((pert_out - input), p=float('inf')) > epsilon / 255:
            break
    
    pert_out = torch.clamp(pert_out, 0, 1)
    adv_pert = pert_out - input

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

def simple_attack(image_folder, model, weights, epsilon, classes, num_classes=3, graph=False, folder=False, control=False):
    """Test function to generate adversarial images and obtain predictions."""
    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Prepare output directory
    if folder:
        output_dir = f"epsilon_{epsilon}"
        os.makedirs(output_dir, exist_ok=True)

    results = []
    print('Iterating over images in folder')
    for image_name in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)

        # Load and transform image
        img = Image.open(image_path)
        transformed_image = weights(img).unsqueeze(0)
        original_image = transformed_image.clone()

        # Calculate gradients
        transformed_image.requires_grad = True
        output = model(normalize(transformed_image))
        target = output.max(1)[1]
        loss = F.nll_loss(output, target)

        model.zero_grad()
        loss.backward()
        data_grad = transformed_image.grad.data

        # Generate adversarial image
        perturbed_image, adv_pert = ifgsm_attack(transformed_image, epsilon, data_grad)

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

            plt.subplot(1, 2, 1)
            plt.imshow(original_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Original: {max(original_dict, key=original_dict.get)} - {max(original_dict.values())}",  fontsize=7)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Perturbed: {max(perturbed_dict, key=perturbed_dict.get)} - {max(perturbed_dict.values())}", fontsize=7)
            plt.axis('off')
            
            plt.subplots_adjust(wspace=0.5)  # Add space between the plots
            plt.show()

    # Save results to txt
    if folder:
        with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
            f.write(f"Epsilon used: {epsilon}\n")
            for result in results:
                f.write(f"Original: {result['original']}\nPerturbed: {result['perturbed']}\n\n")

    return results


#2)-----------------ENSAMBLE ATTACK iFGSM-------------------------
def ensamble_attack(image_folder, models, weights, epsilon, classes, num_classes=3, graph=False, folder=False, control=False):
    """Test function to generate adversarial images and obtain predictions using an ensemble of models."""
    for model in models:
        model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Prepare output directory
    if folder:
        output_dir = f"epsilon_{epsilon}"
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
        
        # Calculate gradients
        target = original_output.max(1)[1]
        loss = F.nll_loss(original_output, target)
        
        for model in models:
            model.zero_grad()
        loss.backward()
        data_grad = transformed_image.grad.data

        # Generate adversarial image
        perturbed_image, adv_pert = ifgsm_attack(transformed_image, epsilon, data_grad)

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

            plt.subplot(1, 2, 1)
            plt.imshow(original_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Original: {max(original_dict, key=original_dict.get)} - {max(original_dict.values())}",  fontsize=7)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Perturbed: {max(perturbed_dict, key=perturbed_dict.get)} - {max(perturbed_dict.values())}", fontsize=7)
            plt.axis('off')
            
            plt.subplots_adjust(wspace=0.5)  # Add space between the plots
            plt.show()

    # Save results to txt
    if folder:
        with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
            f.write(f"Epsilon used: {epsilon}\n")
            for result in results:
                f.write(f"Original: {result['original']}\nPerturbed: {result['perturbed']}\n\n")

    return results
