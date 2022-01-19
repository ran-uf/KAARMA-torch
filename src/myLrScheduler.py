def get_lr(epoch: int) -> float:
    return min(0.05 * epoch + 0.05, 1 * 3. ** ((20 - epoch) / 20) + 0.1)
