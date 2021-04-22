PyTorch Implementation of https://github.com/kmaninis/COB adapted from https://github.com/lejeunel/cobnet

Use method "initRetrain(cfg)" from cobtrain.py to re-train the CobNet, copy PyTorch save to base models after.

import helpers.cob.cobtrain as cobtrain

cfg = {
    'images': "/home/jovyan/work/ma/helpers/cob/pascal-voc/VOC2012",
    'segments': "/home/jovyan/work/ma/helpers/cob/trainval",
    'run': "/home/jovyan/work/runs/cob",
    'lr': 1e-4,
    'decay': 2e-4,
    'momentum': 0.9,
    'epochs-div-lr': 6,
    'epochs': 10,
    'aug-n-angles': 4,
}

cobtrain.initRetrain(cfg)