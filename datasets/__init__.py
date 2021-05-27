from datasets import denmark_points, denmark_shapes, denmark_cob, denmark_all, trancos, voc

def getDataset(name, path, images, object_type, n_classes, transform):
    
    if name == 'denmark_points':
        dataset = denmark_points.Denmark(path, images, object_type, n_classes, transform)
    elif name == 'denmark_shapes':
        dataset = denmark_shapes.Denmark(path, images, object_type, n_classes, transform)
    elif name == 'denmark_all':
        dataset = denmark_all.Denmark(path, images, object_type, n_classes, transform)
    elif name == 'denmark_points_cob':
        dataset = denmark_cob.Denmark(path, images, object_type, n_classes, transform)
    elif name == 'trancos':
        dataset = trancos.Trancos(path, images, object_type, n_classes, transform)
    elif name == 'voc':
        dataset = voc.PascalVOC(path, images)
    else:
        raise ValueError(f'Dataset {name} not defined.')

    return dataset