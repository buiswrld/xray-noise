import torch
import matplotlib.pyplot as plt

from dataset import get_dataloaders

def show_batch(images, labels, title="Batch"):
    images = images[:4]
    labels = labels[:4]
    images = images.cpu().detach()

    if images.shape[1] == 1:
        images = images.squeeze(1)

    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    fig.suptitle(title)

    for i, ax in enumerate(axes):
        img = images[i]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else: # 3 channels
            img = img.permute(1, 2, 0)
            ax.imshow(img)
        ax.set_title(f"label={labels[i].item()}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    root_dir = "../../Downloads/chest_xray"#set your path(i downloaded the dataset and saved locally)

    train_loader, val_loader, test_clean_loader, noisy_loaders = get_dataloaders(
        root_dir=root_dir,
        batch_size=8,
        img_size=224,
        noise_levels=[1, 2]#just for testing
    )

    print("=== TRAIN BATCH (clean) ===")
    images, labels = next(iter(train_loader))
    print("Train batch - images shape:", images.shape)
    print("Train batch - labels shape:", labels.shape)
    print("Labels:", labels)
    show_batch(images, labels, title="Train (clean)")

    print("=== TEST BATCH (clean) ===")
    test_images, test_labels = next(iter(test_clean_loader))
    print("Test batch - images shape:", test_images.shape)
    print("Test batch - labels shape:", test_labels.shape)
    show_batch(test_images, test_labels, title="Test (clean)")

    if noisy_loaders:
        for intensity, loader in noisy_loaders.items():
            print(f"=== TEST BATCH (noisy, intensity={intensity}) ===")
            noisy_imgs, noisy_labels = next(iter(loader))
            print("Noisy images shape:", noisy_imgs.shape)
            show_batch(noisy_imgs, noisy_labels, title=f"Test (noisy, {intensity})")
            break

if __name__ == "__main__":
    main()
