import os
import json
import pickle

def saveJSON(fname, data, makedirs = True):
    dirname = os.path.dirname(fname)
    if makedirs and dirname != '':
        os.makedirs(dirname, exist_ok=True)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)
        

def loadJSON(fname, decode = None):  # TODO: decode???
    with open(fname, "r") as json_file:
        d = json.load(json_file)
    return d


def readText(fname):
    with open(fname, "r", encoding="utf-8", errors='replace') as f:
        lines = f.readlines()
    return lines


def loadPKL(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def savePKL(fname, data, with_rename = True, makedirs = True):
    dirname = os.path.dirname(fname)
    if makedirs and dirname != '':
        os.makedirs(dirname, exist_ok=True)

    # Save file
    if with_rename:
        fname_tmp = fname + "_tmp.pth"
        with open(fname_tmp, "wb") as f:
            pickle.dump(data, f)
        if os.path.exists(fname):
            os.remove(fname)
        os.rename(fname_tmp, fname)
    else:
        with open(fname, "wb") as f:
            pickle.dump(data, f)