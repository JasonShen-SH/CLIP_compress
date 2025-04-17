import clip
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from get_args import get_args
    
class CustomImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        model, _ = clip.load("ViT-L/14@336px", device="cuda")
        self.model = model
        self.device = "cuda"
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        with torch.no_grad():
            sample = sample.unsqueeze(0).to(self.device)
            feature = self.model.encode_image(sample).squeeze(0) 
        return feature, target

def extract_features(dataset):
    features = []
    labels = []
    for i in range(len(dataset)):
        feature, label = dataset[i]
        features.append(feature)
        labels.append(label)
    features = torch.stack(features)
    labels = torch.tensor(labels)
    return features, labels

def get_feature_loaders(img_folder, save_path):
    preprocess = transforms.Compose([
        transforms.Resize(336),
        transforms.CenterCrop((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    dataset = CustomImageDataset(root=f'{img_folder}', transform=preprocess)
    img_features, labels = extract_features(dataset)
    torch.save((img_features, labels), save_path)
    return (img_features, labels)

if __name__ == '__main__':
    """
    In this case, we take Food101 as illustration, change it to whatever benchmark dataset you like.
    """
    # change to your path
    img_folder = "data/food-101/images" # source_folder
    save_path = "data/features/food_features.pt" # save_folder
    img_features, labels = get_feature_loaders(img_folder, save_path)