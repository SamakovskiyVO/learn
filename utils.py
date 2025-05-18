import torch, random, numpy as np

def fix_seed(seed: int = 42):
    """Устанавливаем seed во всех основных библиотеках."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model, trainable_only=True):
    """Подсчитываем количество параметров."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
