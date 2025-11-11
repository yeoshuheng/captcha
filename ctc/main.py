import torch
import argparse
from torch.utils.data import DataLoader
import os
from commons import NUM_CLASSES
from dataset import CaptchaDataset, collate_fn
from model import EnhancedCRNN
from loss import LabelSmoothingCTCLoss
from workflows import evaluate, train_epoch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=64)
    parser.add_argument('--img_width', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}\n")
    
    train_dataset = CaptchaDataset(args.train_dir, args.img_height, args.img_width, augment=True)
    test_dataset = CaptchaDataset(args.test_dir, args.img_height, args.img_width, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    model = EnhancedCRNN(NUM_CLASSES, args.hidden_size, args.img_height, args.img_width)
    model = model.to(device)
    
    criterion = LabelSmoothingCTCLoss(blank=0, smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    best_accuracy = 0.0
    patience = 20
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        train_loss, blank_ratio = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        print(f"train loss: {train_loss:.4f} | blank ratio: {blank_ratio*100:.1f}%")
        
        accuracy, char_acc, avg_conf = evaluate(model, test_loader, device)
        print(f"test acc: {accuracy:.2f}% | char: {char_acc:.2f}% | conf: {avg_conf:.3f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'char_accuracy': char_acc,
                'confidence': avg_conf
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"new best model! acc: {accuracy:.2f}%")
       
    
    print(f"\n{'='*80}")
    print(f"training completed, top acc: {best_accuracy:.2f}%")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()