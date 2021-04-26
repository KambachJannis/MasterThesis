from models import lcfcn, wtp, cob

def getModel(exp_dict, n_classes = None):
    name = exp_dict['model']['name']
    if name in ["lcfcn"]:
        model =  lcfcn.LCFCN(exp_dict, n_classes)
    elif name in ["wtp"]:
        model =  wtp.WTP(exp_dict, n_classes)
    elif name in ["cob"]:
        model =  cob.COB(exp_dict, n_classes)
    else:
        raise ValueError(f'Model {name} not defined.')
    return model