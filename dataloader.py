from torch.utils.data import DataLoader
from dataset import ExteriorData

def get_train_data_loader(root, batchsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = ExteriorData(root,transform=True,phase = 'train')
    train_data_loader = DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return train_data_loader


def get_val_data_loader(root, batchsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = ExteriorData(root,transform=True,phase = 'valid')
    val_data_loader = DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return val_data_loader