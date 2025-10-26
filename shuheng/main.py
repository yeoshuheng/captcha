import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from typing import List
from model import CaptchaModel
from dataset import CHARSET, CaptchaImageDataset, get_image_paths_and_labels
from ctc_trainer import ctc_criterion, CTCTrainer, ctc_preprocess, decode

def train_captcha_model(lr, epochs, train_file_paths: List[str], train_labels: List[str]):
    n_classes = len(CHARSET)

    model = CaptchaModel(n_classes=n_classes)

    dataset = CaptchaImageDataset(train_file_paths, train_labels)

    trainer = CTCTrainer(
        model=model,
        dataset=dataset,
        criterion_function=ctc_criterion,
        optimizer=torch.optim.Adam,
        epochs=epochs,
        lr=lr,
        batch_size=32,
        collate_function=ctc_preprocess,
        checkpoint_interval=5,
        best_model_checkpoint=True,
        gradient_clip=5.0 
    )

    trainer.train()

def test_captcha_model(test_model_ckpt_fp: str, test_file_paths: List[str], test_labels: List[str]):
    n_classes = len(CHARSET)
    
    model = CaptchaModel(n_classes=n_classes)
    model.load_state_dict(torch.load(test_model_ckpt_fp)["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = CaptchaImageDataset(test_file_paths, test_labels)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=ctc_preprocess)

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels_concat, _, target_lengths in dataloader:
            images = images.to(device) 
            
            logits = model(images)
            logits = F.log_softmax(logits, dim=2)
            logits = logits.permute(1, 0, 2)

            print(f"[DEBUG] After log_softmax - shape: {logits.shape}")
            print(f"[DEBUG] After log_softmax - min/max: {logits.min():.4f} / {logits.max():.4f}")

            argmax_vals = logits.argmax(dim=2)
            print(f"[DEBUG] Argmax values shape: {argmax_vals.shape}")
            print(f"[DEBUG] Argmax unique values: {torch.unique(argmax_vals).tolist()}")
            print(f"[DEBUG] First sequence (first 30 timesteps): {argmax_vals[:30, 0].tolist()}")
            print(f"[DEBUG] Count of non-zero predictions: {(argmax_vals != 0).sum().item()} / {argmax_vals.numel()}")
            
            # Check if model is just predicting blank everywhere
            blank_ratio = (argmax_vals == 0).float().mean().item()
            print(f"[DEBUG] Blank token ratio: {blank_ratio:.2%}")


            predicted_captchas = decode(logits.cpu())

            true_labels = []
            start = 0
            for length in target_lengths:
                label_seq = labels_concat[start : start + length]
                chars = [CHARSET[idx - 1] for idx in label_seq]
                true_labels.append("".join(chars))
                start += length

            for pred, true in list(zip(predicted_captchas, true_labels))[:1]:
                print(f"[DEBUG] predicted: {pred}, true: {true}")

            for pred, true in zip(predicted_captchas, true_labels):
                if pred == true:
                    correct += 1
                total += 1

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test CAPTCHA recognition model.")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory containing CAPTCHA images and labels")
    parser.add_argument("--test_dir",  required=True, help="Directory containing CAPTCHA images and labels")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints", help="Path to model checkpoint for testing")
    parser.add_argument("--epochs", type = int, required=True)
    parser.add_argument("--lr", type=float, required=True)

    args = parser.parse_args()

    file_paths, labels = get_image_paths_and_labels(args.train_dir)

    train_captcha_model(args.lr, epochs=args.epochs, train_file_paths=file_paths, train_labels=labels)


    file_paths, labels = get_image_paths_and_labels(args.test_dir)

    test_captcha_model(
        test_model_ckpt_fp=f"{args.checkpoint}/best_model.pth",
        test_file_paths=file_paths,
        test_labels=labels
    )
