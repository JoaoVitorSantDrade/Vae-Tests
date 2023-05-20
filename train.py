import os
import torch.backends.cuda
import torch.backends.cudnn
import torch
import torch.optim as optim
import torchvision.datasets as datasets
from wakepy import keepawake 
from torch.utils.data import DataLoader
from threading import Thread
from datetime import datetime
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
    plot_cnns_tensorboard,
    remove_graphs
)
from model import VAE_GAN, vae_loss, gan_loss
from math import log2
from tqdm import tqdm
import config
import warnings
from torchcontrib.optim import SWA
import copy

def load_tensor(x):
        """
        Carrega um tensor diretamente no Device escolhido.

        Usado em get_loader
        
        :return: o Tensor já carregado no Device
        """
        x = torch.load(x, map_location=f"{config.DEVICE}")
        return x

def get_loader(image_size):
    """
    Retorna um Loader e Um Dataset

    image_size: Dimensão das imagens que vão ser lidas do dataset
    """
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    
    dataset = datasets.DatasetFolder(root=f"Datasets/{config.DATASET}/{image_size}x{image_size}",loader=load_tensor,extensions=['.pt'])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=12,
        persistent_workers=True,
        multiprocessing_context='spawn',
    )
    return loader, dataset

def train_fn(
    model,
    loader,
    epoch,
    opt_gan,
    opt_vae,
    kld_weight,
    tensorboard_step,
    writer,
    scaler_vae,
    scaler_gan,
    scheduler_gen,
    scheduler_disc,
    now,
    ):

    loop = tqdm(loader,position=1, leave=False, unit_scale=True, smoothing=1.0, colour="cyan", ncols=120, desc=f"epoch: {epoch}")
    loop.set_postfix({"VAE Loss": 0, "GAN loss": 0, "KLD Weight": kld_weight})
    model.train()
    train_loss_vae = 1
    train_loss_gan = 1
    for batch_idx, (real, _) in enumerate(loop):
        # imagens por segundo batch_size * it

        if batch_idx == 0:
            first_real = real.detach()

        bundle = (first_real[0],first_real[1])
        opt_gan.zero_grad(set_to_none = True)
        opt_vae.zero_grad(set_to_none = True)

        # Train Disc
        with torch.cuda.amp.autocast(enabled=True):

            # Treinar o VAE
            recon_batch, mu, logvar = model(real)
            vae_loss_val = vae_loss(recon_batch, real, mu, logvar, kld_weight) * ((train_loss_gan*train_loss_gan) + 0.5)
            scaler_vae.scale(vae_loss_val).backward()
            scaler_vae.step(opt_vae)
            scaler_vae.update()

            # Treinar o Discriminador
            z = model.reparameterize(mu, logvar)
            fake_samples = model.decode(z)
            D_real = model.discriminator(real)
            D_fake = model.discriminator(fake_samples.detach())
            gan_loss_val = gan_loss(D_real, D_fake)
            scaler_gan.scale(gan_loss_val).backward()
            scaler_gan.step(opt_gan)
            scaler_gan.update()
        
        loop.set_postfix({"VAE Loss": train_loss_vae, "GAN loss": train_loss_gan, "KLD Weight": kld_weight})
        if batch_idx % 500 == 0:
            train_loss_vae = vae_loss_val.item()
            train_loss_gan = gan_loss_val.item()
            with torch.no_grad():
                fixed_fakes = model(first_real)
                plot_thread = Thread(target=plot_to_tensorboard, args=(writer, train_loss_vae, train_loss_gan, real.detach(), fixed_fakes, tensorboard_step, now,), daemon=True)
                plot_thread.start()
            tensorboard_step += 1

    # Atualiza o peso da divergência KL para a próxima iteração
    if epoch > 50 and epoch < 250:
        kld_weight = min(kld_weight + 0.01, 2)
    else:
        kld_weight = max(kld_weight - 0.0005, 0)

    if config.SCHEDULER:
        scheduler_gen.step()
        scheduler_disc.step()


    if config.OPTMIZER == "SWA":
        opt_gan.swap_swa_sgd()
        opt_vae.swap_swa_sgd()

    
    
    return tensorboard_step, kld_weight, bundle

def main():
    now = datetime.now()
    
    print(f"Versão do PyTorch: {torch.__version__}\nGPU utilizada: {torch.cuda.get_device_name(torch.cuda.current_device())}\nDataset: {config.DATASET}\nData-Horario: {now.strftime('%d/%m/%Y - %H:%M:%S')}")
    print(f"CuDNN: {torch.backends.cudnn.version()}\n")
    tensorboard_step = 0

    model = VAE_GAN(config.IN_CHANNELS,config.Z_DIM,config.LATENT_DIM).to(config.DEVICE)

    #Initialize optmizer and scalers for FP16 Training
    match config.OPTMIZER:
        case "RMSPROP":
            opt_gen = optim.RMSprop(model.parameters(), lr=config.LEARNING_RATE_GENERATOR, weight_decay=config.WEIGHT_DECAY, foreach=True,momentum=0.5)
            opt_disc = optim.RMSprop(model.discriminator.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, weight_decay=config.WEIGHT_DECAY, foreach=True, momentum=0.5)
        case "ADAM":
            opt_gen = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER,weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.Adam(model.discriminator.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER,weight_decay=config.WEIGHT_DECAY)
        case "NADAM":
            opt_gen = optim.NAdam(model.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.NAdam(model.discriminator.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case "ADAMAX":
            opt_gen = optim.Adamax(model.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.Adamax(model.discriminator.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case "ADAMW":
            opt_gen = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.AdamW(model.discriminator.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case "ADAGRAD":
            opt_gen = optim.Adagrad(model.parameters(), lr=config.LEARNING_RATE_GENERATOR,eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.Adagrad(model.discriminator.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR,eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case "SGD":
            opt_gen = optim.SGD(model.parameters(), lr=config.LEARNING_RATE_GENERATOR,momentum=0.9, nesterov=True, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.SGD(model.discriminator.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR,momentum=0.9, nesterov=True, weight_decay=config.WEIGHT_DECAY)
        case "SAM": #https://github.com/davda54/sam https://github.com/mosaicml/composer/tree/dev/composer/algorithms/sam
            #opt_gen = optim.AdamW
            #opt_disc = optim.AdamW
            #opt_gen = sam.SAM(gen.parameters(),opt_gen, lr=config.LEARNING_RATE_GENERATOR, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
            #opt_disc = sam.SAM(disc.parameters(),opt_gen, lr=config.LEARNING_RATE_GENERATOR, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
            raise NotImplementedError("Sam is not implemented")
        case "SWA":
            baseGen = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5,0.99),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            baseDisc = optim.AdamW(model.discriminator.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.99),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_gen = SWA(baseGen,swa_start=10,swa_freq=5,swa_lr=0.05)
            opt_disc = SWA(baseDisc,swa_start=10,swa_freq=5,swa_lr=0.05)
        case "ADAMW8":
            #baseGen = bnb.optim.AdamW8bit(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)
            #baseDisc = bnb.optim.AdamW8bit(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)
            raise NotImplementedError("ADAMW8 do not work on Windows")
        case "RADAM":
            opt_gen = optim.RAdam(model.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.RAdam(model.discriminator.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case _:
            raise NotImplementedError(f"Optim function not implemented")


    #Olhar mais a fundo
    schedulerGen = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_gen,
        T_0=config.PATIENCE_DECAY,
        T_mult=1,
        eta_min=config.MIN_LEARNING_RATE, 
        )
    schedulerDisc = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_disc,
        T_0=config.PATIENCE_DECAY,
        T_mult=1,
        eta_min=config.MIN_LEARNING_RATE, 
        )

    scaler_disc = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        aux = f"{config.WHERE_LOAD}"
        aux = aux.replace(f"{config.DATASET}_","")
        writer = SummaryWriter(f"logs/LacadVae/{config.DATASET}/{aux}")
    else:
        writer = SummaryWriter(f"logs/LacadVae/{config.DATASET}/{now.strftime('%d-%m-%Y-%Hh%Mm%Ss')}")
    
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    cur_epoch = 0
    
    if config.LOAD_MODEL:
        epoch_s = [cur_epoch]
        step_s = [step]
        load_checkpoint(
            config.CHECKPOINT_GEN, model, opt_gen, epoch_s, step_s, schedulerGen, config.WHERE_LOAD
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, model.discriminator, opt_disc, epoch_s, step_s, schedulerDisc, config.WHERE_LOAD
        )
        
        for i in range(step, step_s[0]):
            tensorboard_step += config.PROGRESSIVE_EPOCHS[i]

        step = step_s[0]
        cur_epoch = epoch_s[0] + 1
        tensorboard_step += cur_epoch
    
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        try:
            loader, dataset = get_loader(4*2**step)
        except FileNotFoundError:
            print(f"Could not load file of {4*2**step} size")
            exit(1)
        kld_weight = 0.1
        loop_master = tqdm(range(cur_epoch,num_epochs), position=0, ncols=120,colour='blue', desc=f"VAE-GAN({config.OPTMIZER})-{4*2**step} Training")
        for epoch, _ in enumerate(loop_master):
            tensorboard_step, kld_weight, bundle = train_fn(
                model,
                loader,
                epoch,
                opt_disc,
                opt_gen,
                kld_weight,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_disc,
                schedulerGen,
                schedulerDisc,
                now,
            )
            
            if config.LOAD_MODEL:
                data_save_path = config.WHERE_LOAD
            else:
                data_save_path = config.DATASET + "_"+ now.strftime("%d-%m-%Y-%Hh%Mm%Ss")

            if config.GENERATE_IMAGES and ((epoch+1)%config.GENERATED_EPOCH_DISTANCE == 0) or epoch == (num_epochs-1) and config.GENERATE_IMAGES:
                cpy_model = copy.copy(model)
                img_generator = Thread(target=generate_examples, args=(cpy_model, step, bundle,config.N_TO_GENERATE, (epoch-1), (4*2**step), data_save_path,), daemon=True)
                try:
                    img_generator.start()
                except Exception as err:
                    print(f"Erro: {err}")

            if config.SAVE_MODEL:
                gen_check = Thread(target=save_checkpoint, args=(model, opt_gen, schedulerGen, epoch, step, config.CHECKPOINT_GEN, data_save_path,), daemon=True)
                critic_check = Thread(target=save_checkpoint, args=(model.discriminator, opt_disc, schedulerDisc, epoch, step, config.CHECKPOINT_CRITIC, data_save_path,), daemon=True)
                try:
                    gen_check.start()
                    critic_check.start()
                except Exception as err:
                    print(f"Erro: {err}")

        step += 1
        cur_epoch = 0


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision('medium')
    torch.autograd.emit_nvtx = False
    torch.set_num_threads(6)

    with keepawake(keep_screen_awake=False):
        warnings.filterwarnings("ignore")
        path = config.FOLDER_PATH
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', path, '--bind_all', "--samples_per_plugin", "images=200"])
        url = tb.launch()
        print(f"\n\nTensorboard rodando em {url}")
        main()
