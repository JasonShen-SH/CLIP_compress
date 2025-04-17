from pqvae_shared import feature_compression
from get_args import get_args
from runner import validate_epoch, validate_epoch_shuffle
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from utils import FeatureDataset, logging

device = "cuda"

if __name__ == '__main__':
    args = get_args()
    args.show_bits = True # show the number of bits that have been compressed to

    model = feature_compression(args,device).to(device)
    model.load_state_dict(torch.load(f"latest_models/pqvae_shared/dim:{args.e_dim} size:{args.n_e}.pt", map_location=device), strict=False) # change to your path
    model.to(device)
    
    """
    In this case, we take Food101 as illustration, replace it with any dataset of your choice.
    """

    img_features, labels = torch.load("data/features/food_features.pt", map_location=device)
    dataset = FeatureDataset(img_features, labels)
    dataloader = DataLoader(dataset, batch_size=args.test_batch, shuffle=False)
        
    text_features = torch.load("data/labels/food_text_features.pt", map_location=device)
    
    acc = validate_epoch(model, dataloader, text_features, device)

    logging.info(f"accuracy: {acc}")