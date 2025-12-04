import os
from torch.utils.tensorboard import SummaryWriter

def create_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer
