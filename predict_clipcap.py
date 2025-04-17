import os
import torch
from torch import nn
import torch.nn.functional as F
import skimage.io as io
from PIL import Image
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from get_args import get_args
from clipcap_pqvae import compress_then_caption
import clip

class Predictor():
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        self.clip_model, self.preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.prefix_length = 10
        self.prefix_size = 768
        args = get_args()

        self.models = {}
        for key, weights_path in WEIGHTS_PATHS.items():
            model = compress_then_caption(args, self.device).to(self.device) 
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            model = model.eval()
            model = model.to(self.device)
            self.models[key] = model

    def predict(self, image, model="coco", use_beam_search=False):
        """Run a single prediction on the model"""
        image = io.imread(image)
        model = self.models[model]
        pil_image = Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(self.device, dtype=torch.float32)
            feature_normed = F.normalize(prefix, p=2, dim=1).float()
            encoded_x = model.encoder(feature_normed)
            encoded_x = F.normalize(encoded_x, p=2, dim=1).float()
            _, z_q_reconstructed = model.quantize(encoded_x)
            decoded_x = model.decoder(z_q_reconstructed)
            decoded_x = F.normalize(decoded_x, p=2, dim=1)
            prefix_embed = model.clip_project(decoded_x).reshape(1, self.prefix_length, -1)

        return generate(model, self.tokenizer, embed=prefix_embed)


def generate(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entry_idx in range(entry_count): # entry_count=1
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


if __name__ == '__main__':
    model_path = "/root/autodl-tmp/clipcap_models_pqvae/dim:16 size:4/000-codebook.pt"

    WEIGHTS_PATHS = {
        "coco": model_path,
    }

    predictor = Predictor()
    predictor.setup()

    image_path = "COCO_val2014_000000000831.jpg" # a random image
    generated_list = predictor.predict(image_path)

    print(generated_list)
        