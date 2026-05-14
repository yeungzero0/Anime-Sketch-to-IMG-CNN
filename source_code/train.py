import torch
from config import *
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from piqa import SSIM

def save_examples(generator, val_loader, epoch, folder):
    x, y, ref = next(iter(val_loader)) 
    x, y, ref = x.to(DEVICE), y.to(DEVICE), ref.to(DEVICE)
    generator.eval()
    with torch.no_grad():
        y_fake = generator(x, ref)  # Pass reference image
        y_fake = y_fake * 0.5 + 0.5 # Remove normalization
        save_image(y_fake, folder + f"/colorized_image_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_Sketch_{epoch}.png")
        save_image(y * 0.5 + 0.5, folder + f"/original_Label_{epoch}.png")
        save_image(ref * 0.5 + 0.5, folder + f"/additional_Ref_{epoch}.png")
    generator.train()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, learning_rate):
    print(f"=> Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

def evaluate_model(generator, val_loader):
    ssim = SSIM().to(DEVICE)
    generator.eval()
    total_ssim = 0
    
    with torch.no_grad():
        for input_image, target_image, ref_image in val_loader:
            input_image, target_image, ref_image = input_image.to(DEVICE), target_image.to(DEVICE), ref_image.to(DEVICE)
            
            fake_image = generator(input_image, ref_image)
            fake_image = fake_image * 0.5 + 0.5
            
            target_image = target_image * 0.5 + 0.5
            total_ssim += ssim(fake_image, target_image).item()
    generator.train()
    return total_ssim / len(val_loader)

def train_one_epoch(
    discriminator,
    generator,
    data_loader,
    disc_optimizer,
    gen_optimizer,
    l1_loss,
    bce_loss,
    gen_scaler,
    disc_scaler,
):
    loop = tqdm(data_loader, leave=True)
    for batch_idx, (input_image, target_image, ref_image) in enumerate(loop):
        input_image = input_image.to(DEVICE)
        target_image = target_image.to(DEVICE)
        ref_image = ref_image.to(DEVICE)

        # Train Discriminator
        with torch.amp.autocast('cuda'):
            fake_image = generator(input_image, ref_image)  # Pass reference image
            real_outputs, real_features = discriminator(input_image, target_image, return_features=True)
            fake_outputs, fake_features = discriminator(input_image, fake_image.detach(), return_features=True)
            disc_loss = 0
            for real_out, fake_out in zip(real_outputs, fake_outputs):
                real_loss = bce_loss(real_out, torch.ones_like(real_out))
                fake_loss = bce_loss(fake_out, torch.zeros_like(fake_out))
                disc_loss += (real_loss + fake_loss) / 2
            disc_loss /= len(real_outputs)

        discriminator.zero_grad()
        disc_scaler.scale(disc_loss).backward()
        disc_scaler.step(disc_optimizer)
        disc_scaler.update()

        # Train Generator
        with torch.amp.autocast('cuda'):
            fake_outputs, fake_features = discriminator(input_image, fake_image, return_features=True)
            gen_fake_loss = 0
            for fake_out in fake_outputs:
                gen_fake_loss += bce_loss(fake_out, torch.ones_like(fake_out))
            gen_fake_loss /= len(fake_outputs)
            l1_loss_value = l1_loss(fake_image, target_image) * L1_WEIGHT
            fm_loss = 0
            for real_f, fake_f in zip(real_features, fake_features):
                for rf, ff in zip(real_f, fake_f):
                    fm_loss += nn.L1Loss()(ff, rf.detach())
            fm_loss *= FM_WEIGHT
            gen_loss = gen_fake_loss + l1_loss_value + fm_loss

        generator.zero_grad()
        gen_scaler.scale(gen_loss).backward()
        gen_scaler.step(gen_optimizer)
        gen_scaler.update()

        if batch_idx % 10 == 0:
            loop.set_postfix(
                real_output=torch.sigmoid(real_outputs[0]).mean().item(),
                fake_output=torch.sigmoid(fake_outputs[0]).mean().item(),
                disc_loss=disc_loss.item(),
                gen_loss=gen_loss.item(),
            )