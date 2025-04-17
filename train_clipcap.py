from clipcap_pqvae import compress_then_caption
from pqvae_shared import feature_compression  # import pretrained model
from tqdm import tqdm
import sys
from get_args import get_args
import torch
import torch.nn as nn
from torch.nn import functional as nnf
import torch.optim as optim
from torch.utils.data import DataLoader
from coco_dataset import ClipCocoDataset
from transformers import AdamW, get_linear_schedule_with_warmup
import os


if __name__ == '__main__':
    args = get_args()
    device = "cuda"
    model = compress_then_caption(args, device).to(device)

    # load pretrained (optional)
    model_pretrain = feature_compression(args, device).to(device)  
    model_pretrain.load_state_dict(torch.load(f"/root/autodl-tmp/latest_models/pqvae_shared/dim:{args.e_dim} size:{args.n_e}.pt", map_location=device), strict=False)
    
    model.encoder.load_state_dict(model_pretrain.encoder.state_dict())
    model.decoder.load_state_dict(model_pretrain.decoder.state_dict())
    model.quantize.embedding.weight.data.copy_(model_pretrain.quantize.embedding.weight.data)

    # freeze all feature compression modules (PQVAE-shared)
    for param in (model.encoder.parameters(), model.decoder.parameters(), model.quantize.embedding.parameters()):
        for p in param:
            p.requires_grad = False
 
    model.train()

    # Please download coco2014
    data_path = 'data/coco2014/train_data.pkl' 
    prefix_length = args.prefix_length
    gpt2_type = "gpt2"
    normalize_prefix = False

    dataset = ClipCocoDataset(data_path, prefix_length, gpt2_type, normalize_prefix)
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch, shuffle=True, drop_last=True)

    # start training
    lr = 2e-5
    warmup_steps = 5000
    total_epochs = 10
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_epochs * len(train_dataloader)
    )

    output_dir = "/root/autodl-tmp/clipcap_models_pqvae"
    os.makedirs(output_dir, exist_ok=True)
    subfolder_dir = os.path.join(output_dir, f"dim:{args.e_dim} size:{args.n_e}")
    os.makedirs(subfolder_dir, exist_ok=True)
    
    epoch = 0
    while epoch < total_epochs:
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc="")
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs, codebook_loss = model(tokens, prefix, mask) 
            logits = outputs.logits[:, dataset.prefix_length - 1: -1] # [64, 40, 50257]
            ce_loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)  # logits: [64, 40, 50257];  tokens: [64, 40]
            loss = ce_loss + 10 * codebook_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
        progress.close()
        if epoch % args.save_every == 0 or epoch == total_epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(subfolder_dir, f"{epoch:03d}-codebook.pt"),
            )

        epoch += 1
