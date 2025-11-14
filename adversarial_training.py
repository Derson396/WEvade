import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
from tqdm import tqdm

from utils import get_data_loaders, transform_image, AverageMeter
from model.robust_model import RobustModel


class AdversarialWatermarkLoss(nn.Module):
    """Loss function with adversarial training against WEvade"""
    def __init__(self, alpha=0.5, epsilon=0.01):
        super(AdversarialWatermarkLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha  # balance between quality and robustness
        self.epsilon = epsilon  # attack budget during training
        
    def forward(self, original, watermarked, watermark_gt, decoded_watermark):
        device = decoded_watermark.device
        watermark_gt = watermark_gt.to(device)

        image_loss = self.mse(original, watermarked)
        watermark_loss = self.bce(decoded_watermark, watermark_gt)
        total_loss = image_loss + 0.5 * watermark_loss
        return total_loss

    def adversarial_loss(self, encoder, decoder, original, watermark_gt, criterion, iterations=10, lr=0.1):
        """
        Adversarial attack (PGD-like) robusta — usa updates in-place e loss.backward().
        """
        device = original.device
        watermark_gt = watermark_gt.to(device)

        encoder.train()
        decoder.train()
        encoder.to(device)
        decoder.to(device)

        # Embed inicial (desanexado do grafo de encoder)
        watermarked = encoder(original, watermark_gt).detach().to(device)

        # Versão adversarial (leaf tensor com requires_grad True)
        watermarked_adv = watermarked.clone().detach().to(device)
        watermarked_adv.requires_grad_(True)

        # target aleatório no device correto
        target_watermark = torch.randint(0, 2, watermark_gt.shape, dtype=torch.float32, device=device)

        for _ in range(iterations):
            # garante grad habilitado
            torch.set_grad_enabled(True)

            # Forward
            decoded = decoder(watermarked_adv)
            target = target_watermark.to(decoded.device)

            # Perda escalar
            loss = criterion(decoded, target)

            # Zera grad anterior do tensor adversarial (se houver)
            if watermarked_adv.grad is not None:
                watermarked_adv.grad.detach_()
                watermarked_adv.grad.zero_()

            # Backward: calcula grad wrt watermarked_adv
            loss.backward()

            # Obtém grad (agora em watermarked_adv.grad)
            grad = watermarked_adv.grad
            if grad is None:
                raise RuntimeError("grad is None: decoder might be using torch.no_grad() or detached ops")

            # Update in-place sem quebrar o grafo
            with torch.no_grad():
                # passo FGSM simples (sinal do grad)
                watermarked_adv.add_(-lr * grad.sign())

                # projeta na bola epsilon em relação à imagem original
                perturbation = torch.clamp(watermarked_adv - watermarked, -self.epsilon, self.epsilon)
                watermarked_adv.copy_(torch.clamp(watermarked + perturbation, -1.0, 1.0))

            # garante que requer grad para próxima iteração
            watermarked_adv.requires_grad_(True)

        # Decodifica final (sem grad)
        decoded_adv = decoder(watermarked_adv.detach())

        # Perda robusta: deve recuperar watermark_gt (move para device de decoded_adv)
        robust_loss = self.bce(decoded_adv, watermark_gt.to(decoded_adv.device))

        return robust_loss


def train_robust_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = RobustModel(args.image_size, args.watermark_length, device)
    
    # Load dataset
    train_loader = get_data_loaders(args.image_size, args.dataset_folder)
    
    # Optimizers
    optimizer_enc = optim.Adam(model.encoder.parameters(), lr=args.lr)
    optimizer_dec = optim.Adam(model.decoder.parameters(), lr=args.lr)
    
    # Loss
    criterion = AdversarialWatermarkLoss(alpha=1.0, epsilon=args.epsilon)
    criterion_attack = nn.BCEWithLogitsLoss()
    
    # Training loop
    model.encoder.train()
    model.decoder.train()
    
    for epoch in range(args.epochs):
        losses = AverageMeter()
        robust_losses = AverageMeter()
        bit_accs = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = transform_image(images).to(device)
            
            # Random watermark
            watermark = torch.randint(0, 2, (images.shape[0], args.watermark_length),
                                     dtype=torch.float32, device=device)
            
            # Forward pass
            watermarked = model.encoder(images, watermark)
            decoded = model.decoder(watermarked)
            
            # Standard loss
            loss = criterion(images, watermarked, watermark, decoded)
            
            # Adversarial training every N batches
            if batch_idx % args.adv_freq == 0:
                robust_loss = criterion.adversarial_loss(
                    model.encoder, model.decoder, images, watermark, 
                    criterion_attack, iterations=5, lr=0.05
                )
                total_loss = loss + args.adv_weight * robust_loss
                robust_losses.update(robust_loss.item(), images.shape[0])
            else:
                total_loss = loss
            
            # Backward pass
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            total_loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()
            
            # Metrics
            with torch.no_grad():
                decoded_bits = torch.round(torch.sigmoid(decoded)).cpu().numpy()
                watermark_np = watermark.cpu().numpy()
                bit_acc = 1 - np.abs(decoded_bits - watermark_np).mean()
                
                losses.update(loss.item(), images.shape[0])
                bit_accs.update(bit_acc, images.shape[0])
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'robust_loss': f'{robust_losses.avg:.4f}',
                'bit_acc': f'{bit_accs.avg:.4f}'
            })
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'enc-model': model.encoder.state_dict(),
                'dec-model': model.decoder.state_dict(),
                'optimizer_enc': optimizer_enc.state_dict(),
                'optimizer_dec': optimizer_dec.state_dict(),
            }, f'{args.checkpoint_dir}/robust_model_epoch_{epoch+1}.pth')
    
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train Robust Watermarking Model')
    parser.add_argument('--dataset-folder', default='./dataset/coco/val', type=str)
    parser.add_argument('--checkpoint-dir', default='./ckpt', type=str)
    parser.add_argument('--image-size', default=128, type=int)
    parser.add_argument('--watermark-length', default=30, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epsilon', default=0.02, type=float, help='Attack budget for adversarial training')
    parser.add_argument('--adv-freq', default=3, type=int, help='Adversarial training frequency')
    parser.add_argument('--adv-weight', default=0.3, type=float, help='Weight for adversarial loss')
    parser.add_argument('--save-freq', default=10, type=int)
    
    args = parser.parse_args()
    train_robust_model(args)


if __name__ == '__main__':
    main()