from models.base import fcn8_resnet, fcn8_vgg16, unet

def getBase(base_name, n_classes):
    if base_name == "fcn8_resnet":
        model = fcn8_resnet.FCN8()
    elif base_name == "fcn8_vgg16":
        model = fcn8_vgg16.FCN8_VGG16(n_classes=n_classes)
    elif base_name == "unet":
        model = unet.UNet(n_classes)
    else:
        raise ValueError(f"Base {base_name} does not exist")

    return model