import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from utils import logging
 
def get_logits(image_features, text_features, logit_scale):
    logits_per_image = (logit_scale * image_features @ text_features.T) # batch_size * batch_size
    logits_per_text = (logit_scale * text_features @ image_features.T)  
    return logits_per_image, logits_per_text  


def train_epoch(args, model, train_loader, device):
    assert args.optimizer == "Adam"
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    model.train()
    total_loss = 0.0
    for _, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        feat_decoded, loss = model(images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / len(train_loader)


def validate_epoch(model, test_loader, text_features, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            feat_decoded = model(images) # normalized within model's last step
            text_features = F.normalize(text_features, p=2, dim=1)
            logits_images, _ = get_logits(feat_decoded, text_features.float(), 1)
            probs = logits_images.softmax(dim=1) 
            predicted = torch.argmax(probs, axis=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    logging.info("Accuracy: %.2f%%", acc)
    return acc


def validate_epoch_shuffle(model, test_loader, text_features, device, sorted_test_indices):
    index_map = {number: idx for idx, number in enumerate(sorted_test_indices)}
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            mapped_labels = [index_map[number] for number in labels.tolist() if number in index_map]
            mapped_labels = torch.tensor(mapped_labels).to(device)
            images, labels = images.to(device), labels.to(device)
            feat_decoded = model(images)
            text_features = F.normalize(text_features, p=2, dim=1)
            sorted_text_features = text_features[sorted_test_indices]
            logits_images, _ = get_logits(feat_decoded, sorted_text_features.float(), 1)
            probs = logits_images.softmax(dim=1) 
            predicted = torch.argmax(probs, axis=1)
            total += mapped_labels.size(0)
            correct += (predicted.cpu() == mapped_labels.cpu()).sum().item()
    acc = 100 * correct / total
    logging.info("Accuracy: %.2f%%", acc)
    return acc
