from models import lcfcn, cob, supervised

def getModel(exp_dict, n_classes = None):
    name = exp_dict['model']['name']
    if name in ["lcfcn"]:
        model =  lcfcn.LCFCN(exp_dict, n_classes)
    elif name in ["cob"]:
        model =  cob.COB(exp_dict, n_classes)
    elif name in ["supervised"]:
        model =  supervised.Supervised(exp_dict, n_classes)
    else:
        raise ValueError(f'Model {name} not defined.')
    return model