# Lab 1 – End-to-End Machine Learning Workflow

This README explains all the steps I followed to complete this lab from start to finish.
It covers the environment setup, cloud usage, SSH configuration, data handling, model training, deployment, and evaluation.
All tasks were performed on a cloud-based Ubuntu server and accessed remotely using VS Code.

## Cloud-Based Execution and Deployment

All work for this lab was performed entirely in the cloud using an Ubuntu Virtual Machine hosted on a cloud platform.
No part of the machine learning workflow was executed on my local machine.

VS Code was used only as a remote interface through the Remote SSH extension.
All computations, including data preprocessing, model training, testing, and deployment, were executed directly on the cloud VM.

The machine learning model was trained on the cloud environment and then deployed from the same server.
This ensures consistency between training and deployment environments and reflects real-world ML practices where models are built and deployed on remote infrastructure.

## Environment and Server Setup

I worked on an Ubuntu server running in the cloud and connected to it using VS Code Remote SSH.
Once connected, the VS Code terminal became the server terminal, meaning all commands were executed directly on the VM.

To confirm the connection, the terminal showed a prompt similar to:

    azureuser@server-name:~$

This confirmed that I was working inside the cloud server environment and not locally.

## SSH Key Creation and Usage

To allow the cloud server to securely connect to GitHub, I created a new SSH key directly on the Ubuntu VM.
This key is different from the SSH key used to connect my local machine to the cloud server.

I navigated to the SSH directory:

    cd ~/.ssh

If the directory did not exist, I created it and set the correct permissions:

    mkdir -p ~/.ssh
    chmod 700 ~/.ssh

I then generated a new SSH key:

    ssh-keygen -t ed25519 -C "myemail@example.com"

When prompted for the file name, I provided a custom name to avoid overwriting any existing keys:

    id_ed25519_github

This generated a private key and a public key.
Only the public key was copied and added to GitHub.

To verify the key:

    ls ~/.ssh

To view the public key:

    cat ~/.ssh/id_ed25519_github.pub

I also added the key to the SSH agent:

    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519_github

Finally, I tested the GitHub connection:

    ssh -T git@github.com

## Project Structure and File Navigation

I used standard Linux commands to navigate and manage files:

- ls to list files
- cd folder_name to enter a folder
- cd .. to go back
- pwd to check current directory

To edit Python files directly from the terminal, I used:

    nano file_name.py

or edited them directly inside VS Code.

## Dataset Loading and Understanding

The dataset was loaded on the cloud server using Python libraries such as pandas and numpy.
I inspected the dataset to understand the structure, column names, data types, and missing values.

Key steps included checking dataset shape, viewing sample rows, identifying numerical and categorical features, and checking for missing or invalid values.
This step was important to decide how preprocessing should be handled.

## Data Preprocessing

Before modeling, the data was cleaned and prepared on the cloud environment.
This included handling missing values, converting data types when necessary, and separating features from the target variable.

If required, feature scaling was applied.
The dataset was then split into training and testing sets on the cloud server to ensure fair evaluation.

## Model Selection, Training, and Deployment

A machine learning model was selected based on the problem type.
The model was trained entirely on the cloud server using the training dataset.

After training, the model was saved and deployed within the same cloud environment.
This simulates a production-ready workflow where models are trained and deployed on remote infrastructure.

Keeping training and deployment on the same cloud platform avoids environment mismatch and improves reliability.

## Model Evaluation

After deployment, the model was evaluated using the test dataset.
Appropriate evaluation metrics were used depending on the task.

The evaluation results demonstrated how well the model generalized to unseen data and confirmed that the deployed model performed correctly.

## End-to-End Machine Learning Workflow Summary

This lab followed a complete end-to-end machine learning pipeline executed in the cloud:

1. Cloud server setup and remote access
2. Secure SSH configuration
3. Dataset loading and inspection on the cloud
4. Data preprocessing
5. Cloud-based model training
6. Model deployment on the cloud
7. Model evaluation

Each step was necessary to ensure a correct, reproducible, and realistic ML workflow.

## Conclusion

This lab demonstrates a full end-to-end machine learning pipeline executed entirely in a cloud environment.
Using a cloud VM for training and deployment mirrors real-world ML and data science practices.

All steps were completed successfully, and the model was trained, deployed, and evaluated on the cloud.
