import os
import argparse
import torch
from torch.utils.data import DataLoader
from data import HAM10000, train_val_test_split
from tqdm import tqdm
import wandb

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--image_dir", default="dataset/imgs", type=str)
parser.add_argument("--mask_dir", default="dataset/masks", type=str)
parser.add_argument("--gpu_num", default="0,1", type=str)
parser.add_argument("--use_wandb", default=True, type=bool)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_wandb:
        wandb.init(project="ham_segmentation")

    ham = HAM10000(image_dir=args.image_dir, mask_dir=args.mask_dir)
    train, val, test = train_val_test_split(dataset=ham, ratio=0.1)

    train_loader = DataLoader(
        train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    val_loader = DataLoader(
        val, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True
    )

    for epoch in range(args.epochs):
        for iter, (X, y) in tqdm(enumerate(train_loader)):
            X, y = X.to(device), y.to(device)
            pass


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
