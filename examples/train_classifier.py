"""
Training Example with dHT

This script demonstrates how to train a classification model with dHT tokenization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse

from dht.nn.transformer import dHTClassifier


def create_dummy_dataset(n_samples=1000, n_classes=10):
    """Create a dummy dataset for demonstration."""
    images = torch.randn(n_samples, 3, 224, 224)
    labels = torch.randint(0, n_classes, (n_samples,))
    return TensorDataset(images, labels)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {100. * correct / total:.2f}%")
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print(f"\nCreating datasets...")
    train_dataset = create_dummy_dataset(args.train_samples, args.n_classes)
    val_dataset = create_dummy_dataset(args.val_samples, args.n_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print(f"\nCreating model...")
    model = dHTClassifier(
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        heads=args.heads,
        depth=args.depth,
        n_classes=args.n_classes,
        channels=3,
        compute_grad=True,
        dop_path=args.dropout,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr / 100
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, args.save_path)
            print(f"  Saved best model (val_acc: {val_acc:.2f}%)")
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train dHT classification model')
    
    # Data
    parser.add_argument('--train-samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=200, help='Number of validation samples')
    parser.add_argument('--n-classes', type=int, default=10, help='Number of classes')
    
    # Model
    parser.add_argument('--embed-dim', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size')
    parser.add_argument('--heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--depth', type=int, default=6, help='Number of transformer blocks')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    
    # Other
    parser.add_argument('--save-path', type=str, default='dht_checkpoint.pth', help='Model save path')
    
    args = parser.parse_args()
    main(args)
