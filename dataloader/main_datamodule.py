import pytorch_lightning as pl
from torch.utils.data import DataLoader

from core.registry import REG


class MainDataModule(pl.LightningDataModule):
    def __init__(self, dataset_config):
        super().__init__()
        self.config = dataset_config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = REG.build('dataset', self.config, self.config.train.name)
            self.val_dataset = REG.build('dataset', self.config, self.config.val.name)
        if stage in ['validate']:
            self.val_dataset = REG.build('dataset', self.config, self.config.val.name)
        if stage in ['test']:
            self.test_dataset = REG.build('dataset', self.config, self.config.test.name)
        if stage in ['predict']:
            self.predict_datasets = dict()
            for predict_target in self.config.predict_targets:
                self.predict_datasets[predict_target] = REG.build('dataset', self.config, self.config[predict_target].dataset_name)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, pin_memory=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=None, pin_memory=True, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=None, pin_memory=True, num_workers=4, persistent_workers=True)

    def predict_dataloader(self):
        predict_dataloaders = []
        for predict_target in self.config.predict_targets:
            predict_dataloaders.append(DataLoader(self.predict_datasets[predict_target], batch_size=None, pin_memory=True, num_workers=4, persistent_workers=True))
        return predict_dataloaders
