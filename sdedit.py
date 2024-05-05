from unittest import result
import torch
import argparse
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
import os
from PIL import Image
from torchvision import transforms
import tqdm
from tqdm import tqdm


def set_timesteps(scheduler, num_inference_steps, strength):
    scheduler.set_timesteps(num_inference_steps)
    new_num_inference_steps = int(num_inference_steps * strength)
    final_timestep = scheduler.timesteps[-new_num_inference_steps]
    t_start = num_inference_steps - new_num_inference_steps
    timesteps = scheduler.timesteps[t_start:].to(device)
    return timesteps, final_timestep


#get the text embedding
def get_embedding_for_prompt(prompt, tokenizer, text_encoder, device, max_len):
    max_length = max_len
    tokens = tokenizer([prompt],
                       padding="max_length",
                       max_length=max_length,
                       truncation=True,
                       return_tensors="pt")
    with torch.no_grad():
        embeddings = text_encoder(tokens.input_ids.to(device))[0]
    return embeddings


def get_text_embedding(tokenizer,
                       text_encoder,
                       device,
                       max_len,
                       text1,
                       text2=''):
    emb = get_embedding_for_prompt(text1, tokenizer, text_encoder, device,
                                   max_len)
    uncond = get_embedding_for_prompt(text2, tokenizer, text_encoder, device,
                                      max_len)
    text_embeddings = torch.cat([uncond, emb]).to(device)
    return text_embeddings


def latents2images(latents):
    latents = latents / 0.18215
    with torch.no_grad():
        imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype("uint8")
    imgs = [Image.fromarray(i) for i in imgs]
    return imgs


def image2latent(image):
    image = transforms.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(image.to(device) * 2 - 1)
    latent = latent.latent_dist.sample() * 0.18215
    return latent


def get_data_pair(file_path):
    pairs = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            list = line.split('<>')
            image_id = list[0]
            prompt = list[2]
            pairs.append([image_id, prompt])
        f.close()
    return pairs


#set seed
torch.manual_seed(1)
#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4").to(device)
vae = pipe.vae
tokenizer = pipe.tokenizer
unet = pipe.unet
scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                                 beta_end=0.012,
                                 beta_schedule="scaled_linear",
                                 num_train_timesteps=1000)

NUM_INFERENCE_STEPS = 50
EMBEDDING_LEN = min(tokenizer.model_max_length,
                    pipe.text_encoder.config.max_position_embeddings)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default='.\\action_change_dataset\dataset.txt',
                        help="text file containing image paths and prompts")
    parser.add_argument("--data_path",
                        type=str,
                        default='.\\action_change_dataset\data',
                        help="folder to load data")
    parser.add_argument("--output_path",
                        type=str,
                        default='.\output',
                        help="folder to save output")
    parser.add_argument("--save_original",
                        type=bool,
                        default=True,
                        help="save original image")
    parser.add_argument("--strength",
                        type=float,
                        default=0.5,
                        help="strength of diffusion")
    args = parser.parse_args()
    return args


@torch.no_grad()
def main(args):
    result_path = os.path.join(args.output_path, 'result_image')
    os.makedirs(result_path, exist_ok=True)
    pairs = get_data_pair(args.input_path)
    for i in pairs:
        img_path = os.path.join(args.data_path, f'{i[0]}.jpg')
        image = Image.open(img_path).resize((512, 512)).convert('RGB')
        prompt = i[1]
        if args.save_original:
            os.makedirs(os.path.join(args.output_path, 'ref_image'),
                        exist_ok=True)
            image.save(
                os.path.join(os.path.join(args.output_path, 'ref_image'),
                             f'{i[0]}.jpg'))
        latent = image2latent(image)
        text_embeddings = get_text_embedding(tokenizer, pipe.text_encoder,
                                             device, EMBEDDING_LEN, prompt)
        timesteps, final_timestep = set_timesteps(scheduler,
                                                  NUM_INFERENCE_STEPS,
                                                  args.strength)

        noise = torch.randn_like(latent)
        final_timestep = torch.tensor([final_timestep], device=device)
        noisy_latent = scheduler.add_noise(latent, noise,
                                           final_timestep).to(device)
        bar = tqdm(timesteps)
        for t in bar:
            latent_model_input = torch.cat([noisy_latent] * 2)
            latent_model_input = scheduler.scale_model_input(
                latent_model_input, t)
            with torch.no_grad():
                noise_pred = unet(latent_model_input.to(unet.dtype),
                                  t,
                                  encoder_hidden_states=text_embeddings.to(
                                      unet.dtype))["sample"]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text -
                                                    noise_pred_uncond)
            noisy_latent = scheduler.step(noise_pred, t,
                                          noisy_latent).prev_sample
            bar.update(1)
        latents = noisy_latent
        result = latents2images(latents)[0]
        result.save(os.path.join(result_path, f'{i[0]}.jpg'))


if __name__ == '__main__':
    args = get_args()
    main(args)
