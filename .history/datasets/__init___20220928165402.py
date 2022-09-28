from datasets.oulad import OULADDataset
from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset
from.oulad import OULADDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML20MDataset.code(): ML20MDataset,
    OULADDataset.code(): OULADDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
