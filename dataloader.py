import json
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class BeerDataset(Dataset):
    def __init__(self, json_path, images_dir, transform=None, label_to_idx=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.images_dir = images_dir
        self.transform = transform

        if label_to_idx is None:
            labels = sorted(set(item["label"] for item in self.data))
            self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.images_dir, item["image"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.label_to_idx[item["label"]]
        return image, label


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def get_dataloaders(data_dir="data", batch_size=32, num_workers=2):
    images_dir = os.path.join(data_dir, "processed_images")

    train_dataset = BeerDataset(
        json_path=os.path.join(data_dir, "train.json"),
        images_dir=images_dir,
        transform=get_transforms(),
    )

    label_to_idx = train_dataset.label_to_idx

    val_dataset = BeerDataset(
        json_path=os.path.join(data_dir, "val.json"),
        images_dir=images_dir,
        transform=get_transforms(),
        label_to_idx=label_to_idx,
    )

    test_dataset = BeerDataset(
        json_path=os.path.join(data_dir, "test.json"),
        images_dir=images_dir,
        transform=get_transforms(),
        label_to_idx=label_to_idx,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, label_to_idx
