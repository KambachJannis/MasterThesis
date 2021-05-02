from datasets import denmark, denmark_cob, denmark_shapes

def getDataset(name, path, images, n_classes, transform):
    
    if name == 'denmark_points':
        dataset = denmark.Denmark(path, images, n_classes, transform)
    elif name == 'denmark_shapes':
        dataset = denmark_shapes.Denmark(path, images, n_classes, transform)
    elif name == 'denmark_cob':
        dataset = denmark_cob.Denmark(path, images, n_classes, transform)
    else:
        raise ValueError(f'Dataset {name} not defined.')

    return dataset