from gcn_model.utils import get_dataloaders

train_loader, valid_loader = get_dataloaders(
    "train_data/combined/train.json",
    "train_data/combined/valid.json",
    batch_size=32
)
