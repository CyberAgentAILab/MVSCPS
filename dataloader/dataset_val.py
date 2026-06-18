from core.registry import REG

from .dataset_eval_base import DiligentEvalDataset


@REG.register("dataset", name='dataset_val')
class DiligentValDataset(DiligentEvalDataset):
    stage = 'val'
    key_prefix = 'val_'
