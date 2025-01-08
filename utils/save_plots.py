import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def output_scatter_plot(scan, mask, epoch, image, part):
    for i, (img, msk) in enumerate(zip(scan, mask)):
        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(1, 2, 1)
        plt.imshow(transforms.ToPILImage()(img), cmap="gray")
        fig.add_subplot(1, 2, 2)
        plt.imshow(transforms.ToPILImage()(msk), cmap="gray")

        plt.savefig(
            f"results/{'test' if epoch is None else 'train'}/{epoch}-{image}-{part}-{i}.png"
        )
        plt.clf()
