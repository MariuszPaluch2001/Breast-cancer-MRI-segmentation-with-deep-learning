import torch
import torch.nn.functional as F


def adjust_tensor(data, mask):
    _, _, D, H, W = data.shape

    assert H % 2 == 0 and W % 2 == 0, "Wymiary H i W muszą być podzielne przez 2."

    yield data[:, :, : D // 2, : H // 2, : W // 2], mask[
        :, :, : D // 2, : H // 2, : W // 2
    ]

    yield data[:, :, : D // 2, : H // 2, W // 2 :], mask[
        :, :, : D // 2, : H // 2, W // 2 :
    ]

    yield data[:, :, : D // 2, H // 2 :, : W // 2], mask[
        :, :, : D // 2, H // 2 :, : W // 2
    ]

    yield data[:, :, : D // 2, H // 2 :, W // 2 :], mask[
        :, :, : D // 2, H // 2 :, W // 2 :
    ]

    yield data[:, :, D // 2 :, : H // 2, : W // 2], mask[
        :, :, D // 2 :, : H // 2, : W // 2
    ]

    yield data[:, :, D // 2 :, : H // 2, W // 2 :], mask[
        :, :, D // 2 :, : H // 2, W // 2 :
    ]

    yield data[:, :, D // 2 :, H // 2 :, : W // 2], mask[
        :, :, D // 2 :, H // 2 :, : W // 2
    ]

    yield data[:, :, D // 2 :, H // 2 :, W // 2 :], mask[
        :, :, D // 2 :, H // 2 :, W // 2 :
    ]
