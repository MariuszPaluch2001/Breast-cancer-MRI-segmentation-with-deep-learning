import torch


def get_tp(output, target):
    return (output[target == 1] == 1).sum().item()


def get_tn(output, target):
    return (output[target == 0] == 0).sum().item()


def get_fp(output, target):
    return (output[target == 0] == 1).sum().item()


def get_fn(output, target):
    return (output[target == 1] == 0).sum().item()


def get_dice(output, target):
    tp = get_tp(output, target)
    fp = get_fp(output, target)
    fn = get_fn(output, target)
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    return 2 * tp / (2 * tp + fn + fp)


def get_precission(output, target):
    tp = get_tp(output, target)
    fp = get_fp(output, target)
    if tp == 0 and fp == 0:
        return 0.0
    return tp / (tp + fp)


def get_recall(output, target):
    tp = get_tp(output, target)
    fn = get_fn(output, target)
    if tp == 0 and fn == 0:
        return 0.0
    return tp / (tp + fn)


def evaluate(output, target):
    print(
        f"Dice: {get_dice(output, target):.4f}, Precission: {get_precission(output, target):.4f}, Recall: {get_recall(output, target):.4f}"
    )


if __name__ == "__main__":
    output = torch.tensor(
        [
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ],
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
        ]
    )

    target = torch.tensor(
        [
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
            ],
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ]
    )

    evaluate(output, target)
