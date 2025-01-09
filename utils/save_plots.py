import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os


def output_scatter_plot(scan, mask, epoch, image, part, channel_label):
    for i, (img, msk) in enumerate(zip(scan, mask)):
        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(1, 2, 1)
        plt.imshow(transforms.ToPILImage()(img), cmap="gray")
        fig.add_subplot(1, 2, 2)
        plt.imshow(transforms.ToPILImage()(msk), cmap="gray")

        os.makedirs(f"results/{channel_label}/{'test' if epoch is None else 'train'}/epoch_{epoch}/image_{image}/part_{part}", exist_ok=True)
        plt.savefig(
            f"results/{channel_label}/{'test' if epoch is None else 'train'}/epoch_{epoch}/image_{image}/part_{part}/slice_{i}.png"
        )
        plt.clf()
