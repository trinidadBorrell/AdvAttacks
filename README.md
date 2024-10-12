### 🛡️ Adversarial Attacks on CNN Ensembles

#### 📚 Description

This repository focuses on generating **adversarial attacks** on an ensemble of **Convolutional Neural Networks (CNNs)**. The ensemble includes the following models: **ResNet-152**, **ResNet-101**, **ResNet-50**, **EfficientNet-B4**, **EfficientNet-B5**, and **Inception-V3**.

#### Attack Methods

We implement both **targeted** and **untargeted** adversarial attacks using **iFSGM** and **FSGM**, considering **thin** and **coarse** class groups. Additionally, we generate **control images** by manipulating the adversarial perturbations through the following methods: *i) Flipping the perturbations*, *ii) Randomizing the phase in Fourier image space*

#### 📂 Repository Structure

- **`utils.py`**: Contains the core functions for generating and applying adversarial attacks.
- **`plottools.py`**: Provides plotting tools to automate the analysis of attacks.
- **Notebooks**: Includes initial analysis and exploratory work on the adversarial attacks and control methods.

#### 🔄 Future Updates

This repository is still being refined, and future updates will focus on enhancing usability and making it more user-friendly. For now, the repository provides the initial structure and tools used to generate over 900 attacked images to run forced-choice experiments in humans.
