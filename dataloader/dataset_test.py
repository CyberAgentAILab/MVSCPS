from core.registry import REG

from .dataset_eval_base import DiligentEvalDataset


@REG.register("dataset", name='dataset_test')
class DiligentTestDataset(DiligentEvalDataset):
    stage = 'test'
    key_prefix = ''
