from datasets import trancos
from datasets import denmark

def getDataset(dataset, split, datadir, transform, dataset_size = None):
    name = dataset
    
    if name == 'trancos':
        dataset = trancos.Trancos(split, datadir, transform)
        if dataset_size is not None and dataset_size[split] != 'all':
            dataset.img_names = dataset.img_names[:dataset_size[split]]
    
    elif name == 'denmark':
        dataset = denmark.Denmark(split, datadir, transform)
        if dataset_size is not None and dataset_size[split] != 'all':
            dataset.img_names = dataset.img_names[:dataset_size[split]]
    
    else:
        raise ValueError(f'Dataset {name} not defined.')

    return dataset