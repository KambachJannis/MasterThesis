from models import lcfcn, wtp, cob

def getModel(model_dict, exp_dict = None, train_set = None):
    name = model_dict['name']
    if name in ["lcfcn"]:
        model =  lcfcn.LCFCN(exp_dict, train_set = train_set)
    elif name in ["wtp"]:
        model =  wtp.WTP(exp_dict, train_set = train_set)
    elif name in ["cob"]:
        model =  cob.COB(exp_dict, train_set = train_set)
    else:
        raise ValueError(f'Model {name} not defined.')
    return model