from models import resnet, vgg16, unet, resunet

def getNet(net_name, n_classes):
    if net_name == "lcfcn":
        model = resnet.ResNet(n_classes-1)
    elif net_name == "vgg16":
        model = vgg16.VGG16(n_classes-1)
    elif net_name == "unet":
        model = resunet.ResNetUNet(n_classes)
    else:
        raise ValueError(f"Net {base_name} does not exist")

    return model