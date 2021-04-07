from models.base import fcn8_resnet, fcn8_vgg16, wtp_vgg16

def getBase(base_name, exp_dict, n_classes):
    if base_name == "fcn8_resnet":
        model = fcn8_resnet.FCN8()
    elif base_name == "fcn8_vgg16":
        model = fcn8_vgg16.FCN8_VGG16(n_classes=n_classes)
    elif base_name == "wtp_vgg16":
        model = wtp_vgg16.WTP_VGG16(n_classes=n_classes)
    else:
        raise ValueError(f"Base {base_name} does not exist")

    return model