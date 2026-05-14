import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from config import *
from discriminator import MultiScaleDiscriminator
from generator import Generator
from train import save_checkpoint, load_checkpoint, train_one_epoch, save_examples, evaluate_model

"""
Please edit the config.py, if you are the first play at this program, 
Ensure LOAD_MODEL is False, it look like (line 45):     LOAD_MODEL = False
After program created the "Disc.pth.tar" & "Gen.pth.tar", ensure LOAD_MODEL is Ture.

For creating your own dataset, save the image at /dataset/newIMG/original
Then run "part1_imageResizeTo512png.py" & "part2_imageToSketch.py"
*dataset should at less 500 more IMG, if you want it be more specific way, you should just create it specifically)
*Just like you want to recolor who anime character you want, you should just download which character as the original dataset.
*Then the sketch will recolored to your ref.

When you want to play the Sketch to IMG, run "main_runThisForTesting(SketchToImage).py"
"""

def main():
    # Initialize models, optimizers, and loss functions
    discriminator = MultiScaleDiscriminator(in_channels=3, scales=DISC_SCALES).to(DEVICE)
    generator = Generator(in_channels=6, features=64).to(DEVICE)  
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    gen_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
        
    # Learning rate schedulers
    disc_scheduler = StepLR(disc_optimizer, step_size=15, gamma=0.5)
    gen_scheduler = StepLR(gen_optimizer, step_size=15, gamma=0.5)

    # Load checkpoints if specified
    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT_GEN, generator, gen_optimizer, LEARNING_RATE)
        load_checkpoint(CHECKPOINT_DISC, discriminator, disc_optimizer, LEARNING_RATE)

    # Create datasets and data loaders
    # train
    train_dataset = ImageDataset(
        input_dir=TRAIN_DIR_SKETCH, 
        target_dir=TRAIN_DIR_REAL, 
        ref_dir=TRAIN_DIR_REF
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # val
    val_dataset = ImageDataset(
        input_dir=VAL_DIR_SKETCH, 
        target_dir=VAL_DIR_REAL, 
        ref_dir=VAL_DIR_REF
        #, sample_size=500
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=RANDOM,
        num_workers=4,
        pin_memory=True
    )
    
    # Gradient scalers for mixed precision training
    gen_scaler = torch.cuda.amp.GradScaler()
    disc_scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        if SAVE_MODEL:
            train_one_epoch(
                discriminator,
                generator,
                train_loader,
                disc_optimizer,
                gen_optimizer,
                l1_loss,
                bce_loss,
                gen_scaler,
                disc_scaler,
            )

        # Step learning rate schedulers
        disc_scheduler.step()
        gen_scheduler.step()

        # Save model checkpoints every 2 epochs
        if SAVE_MODEL and epoch % 2 == 0:
        
        # # Save model checkpoints every epoch
        # if SAVE_MODEL:
            save_checkpoint(generator, gen_optimizer, filename=CHECKPOINT_GEN)
            save_checkpoint(discriminator, disc_optimizer, filename=CHECKPOINT_DISC)

        # Save example outputs and compute SSIM
        save_examples(generator, val_loader, epoch, folder= os.path.join(SCPATH,  'result'))
        # if SAVE_MODEL:
        #     ssim_score = evaluate_model(generator, val_loader)
        #     print(f"Epoch {epoch+1}/{NUM_EPOCHS}, SSIM: {ssim_score:.4f}")

if __name__ == "__main__":
    main()
    print(f"\n\n=======================================================================================")
    print(f"\nmain_runThis.py program is completed! \nIf you want to train the model again, please change SAVE_MODEL be True at config.py\n")
    print(f"=======================================================================================")
    
    
    