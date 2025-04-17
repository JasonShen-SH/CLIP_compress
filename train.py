from pqvae_shared import feature_compression
from get_args import get_args
from runner import train_epoch, validate_epoch, validate_epoch_shuffle
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import os
import numpy as np
from utils import FeatureDataset, logging

device = "cuda"

if __name__ == '__main__':
    args = get_args()
    model = feature_compression(args, device).to(device)
    
    # if args.load_pretrained: # you could either load pretrained model or train from scratch, it's up to you!
    #     model.load_state_dict(torch.load(f"latest_models/pqvae_shared/dim:{args.e_dim} size:{args.n_e}.pt"), strict=False)
    #     model.to(device)
    
    """
    We utilized the ImageNet dataset, with 800 randomly selected classes allocated for training and the remaining 200 classes reserved for validation. 
    For further details, please refer to the paper.
    """

    full_features, full_labels = torch.load("data/features/imagenet_features.pt", map_location = device) 
    full_dataset = FeatureDataset(full_features, full_labels)
    train_indices = np.load("data/indices/shuffled_train_indices.npy")
    test_indices = np.load("data/indices/shuffled_test_indices.npy")
    train_sample_indices = np.concatenate([np.arange(start=class_idx * args.num_samples_per_class, stop=(class_idx + 1) * args.num_samples_per_class) for class_idx in train_indices])
    test_sample_indices = np.concatenate([np.arange(start=class_idx * args.num_samples_per_class, stop=(class_idx + 1) * args.num_samples_per_class) for class_idx in test_indices])
    train_dataset = Subset(full_dataset, sorted(train_sample_indices))
    test_dataset = Subset(full_dataset, sorted(test_sample_indices))
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

    # train and validate
    epoch = 0
    text_features = torch.load("data/labels/text_features.pt", map_location = device)

    # save path
    model_save_path = "latest_models/pqvae_shared_finetune"
    os.makedirs(model_save_path, exist_ok=True)

    best_acc = 0
    test_indices = np.load("data/indices/shuffled_test_indices.npy")
    sorted_test_indices = sorted(test_indices)

    while epoch < args.epoch:
        loss = train_epoch(args, model, train_loader, device)
        logging.info(f'Epoch {epoch}, Loss: {loss:.4f}')

        acc = validate_epoch_shuffle(model, test_loader, text_features, device, sorted_test_indices)

        if acc > best_acc:
            torch.save(model.state_dict(), os.path.join(model_save_path, f"e_dim:{args.e_dim} size:{args.n_e}.pt"))
            best_acc = acc
            
        epoch += 1

    logging.info(f"highest_accuracy is {best_acc}, e_dim:{args.e_dim}, n_e:{args.n_e}")