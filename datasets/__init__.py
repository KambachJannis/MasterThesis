from datasets import denmark_points, denmark_cob, denmark_shapes, trancos

def getDataset(name, path, images, object_type, n_classes, transform):
    
    if name == 'denmark_points':
        dataset = denmark_points.Denmark(path, images, object_type, n_classes, transform)
    elif name == 'denmark_shapes':
        dataset = denmark_shapes.Denmark(path, images, object_type, n_classes, transform)
    elif name == 'denmark_points_cob':
        dataset = denmark_cob.Denmark(path, images, object_type, n_classes, transform)
    elif name == 'trancos':
        dataset = trancos.Trancos(path, images, object_type, n_classes, transform)
    else:
        raise ValueError(f'Dataset {name} not defined.')

    return dataset