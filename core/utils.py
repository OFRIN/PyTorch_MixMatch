import cv2
import torch
import numpy as np

def get_one_hot_vector(label, classes):
    vector = np.zeros((classes), dtype = np.float32)
    if len(label) > 0:
        vector[label] = 1.
    return vector

def get_learning_rate_from_optimizer(optimizer):
    return optimizer.param_groups[0]['lr']

def get_numpy_from_tensor(tensor):
    return tensor.cpu().detach().numpy()

def load_model(model, model_path):
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path))

def save_model(model, model_path):
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

def get_labels_from_dataset(dataset):
    try:
        labels = dataset.targets
    except:
        labels = [label for image, label in dataset]

    return np.asarray(labels)

def split_train_and_validation_datasets(dataset, classes, ratio=0.1):
    labels = get_labels_from_dataset(dataset)
    
    train_indices = []
    validation_indices = []

    for class_index in range(classes):
        indices = np.where(labels == class_index)[0]
        validation_size_per_class = int(len(indices) * ratio)
        
        np.random.shuffle(indices)
        
        train_indices.extend(indices[:-validation_size_per_class])
        validation_indices.extend(indices[-validation_size_per_class:])
    
    return train_indices, validation_indices

def split_labeled_and_unlabeled_datasets(dataset, classes, the_number_of_label_per_class):
    labels = np.asarray([label for image, label in dataset])
    
    labeled_indices = []
    unlabeled_indices = []

    for class_index in range(classes):
        indices = np.where(labels == class_index)[0]

        np.random.shuffle(indices)

        labeled_indices.extend(indices[:the_number_of_label_per_class])
        unlabeled_indices.extend(indices[the_number_of_label_per_class:])

    return labeled_indices, unlabeled_indices
