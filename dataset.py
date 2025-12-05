import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from noise import add_poisson, add_gaussian


class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None,
                 poisson_intensity=0, gaussian_intensity=0):

        self.root_dir = os.path.join(root_dir, split)
        self.dataset = ImageFolder(self.root_dir)
        self.transform = transform

        self.poisson_intensity = poisson_intensity
        self.gaussian_intensity = gaussian_intensity

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.poisson_intensity > 0:
            img = add_poisson(img, intensity=self.poisson_intensity)
        if self.gaussian_intensity > 0:
            img = add_gaussian(img, intensity=self.gaussian_intensity)

        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms(img_size=224):
    """
    Returns (train_transform, val_transform, test_transform)
    for a 1-channel CNN on chest X-rays. <- needs to be changed depedning on a cnn model we're using
    now i used 1-channel bc chest x-rays are grayscale images
    """
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_transform = eval_transform
    test_transform = eval_transform

    return train_transform, val_transform, test_transform


def get_dataloaders(root_dir, batch_size=32, img_size=224, noise_levels=None):
    train_tf, val_tf, test_tf = get_transforms(img_size=img_size)

    train_ds = ChestXrayDataset(root_dir, split="train", transform=train_tf)
    val_ds   = ChestXrayDataset(root_dir, split="val",   transform=val_tf)
    test_ds  = ChestXrayDataset(root_dir, split="test",  transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_clean_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    noisy_loaders = {}
    if noise_levels is not None:
        for nl in noise_levels:
            noisy_ds = ChestXrayDataset(
                root_dir,
                split="test",
                transform=test_tf,
                poisson_intensity=nl,
                gaussian_intensity=0,
            )
            noisy_loaders[nl] = DataLoader(noisy_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_clean_loader, noisy_loaders
