import torch
import clip
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pqvae_shared import feature_compression
from get_args import get_args
from PIL import Image
import os

# Initialize SAM model,  make sure you download sam checkpoint from Github
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) 

sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Initialize CLIP model
clip_model, preprocess = clip.load("ViT-L/14@336px", device=device, jit=False)

# Load image
image = cv2.imread("COCO_val2014_000000000831.jpg")  # I use a random image from COCO-2014 val-set
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB 

# Generate masks using SAM
masks = mask_generator.generate(image)

# Extract text feature
text = "the brown building"
text_tokens = clip.tokenize([text]).to(device)
with torch.no_grad():
    text_feature = clip_model.encode_text(text_tokens)
    text_feature /= text_feature.norm(dim=-1, keepdim=True)  # Normalize

# Process each mask
best_patch = None
best_similarity = -1

# Feature compression using PQVAE-shared
args = get_args()
feature_compress = feature_compression(args, device).to(device)
feature_compress.load_state_dict(torch.load(f"latest_models/pqvae_shared/dim:{args.e_dim} size:{args.n_e}.pt"), strict=False)
feature_compress.to(device)
# choose your option, e.g.(e_dim:16, n_e:4)→400bits, (e_dim:16, n_e:8)→600bits

for i, mask in enumerate(masks):
    # Get the bbox coordinates from the mask
    y, x = np.where(mask['segmentation'])
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # Create a black image with the size of the bbox
    bbox_image = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    
    # Copy only the mask region from the original image to the bbox image
    mask_region = mask['segmentation'][y_min:y_max, x_min:x_max]
    bbox_image[mask_region] = image[y_min:y_max, x_min:x_max][mask_region]
    
    # Preprocess and encode the bbox image with CLIP
    preprocessed_image = preprocess(Image.fromarray(bbox_image)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feature = clip_model.encode_image(preprocessed_image)
        img_feature /= img_feature.norm(dim=-1, keepdim=True)  # Normalize

    # compress feature using PQVAE-shared
    compressed_img_feature, _ = feature_compress(img_feature)

    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(text_feature, compressed_img_feature)
    
    # Track the best matching patch
    if similarity > best_similarity:
        best_similarity = similarity
        best_patch = bbox_image

# Save the best matching patch
if best_patch is not None:
    best_patch = cv2.cvtColor(best_patch, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    cv2.imwrite("best_matching_patch.jpg", best_patch)
    print(f"Best matching patch saved with similarity score: {best_similarity.item():.4f}")
else:
    print("No valid patches found.")