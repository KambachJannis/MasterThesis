import os
import json
import copy
import torch
import shutil
import pickle
import hashlib
import itertools

def deleteExperiment(savedir, backup_flag = False):
    """Delete an experiment. If the backup_flag is true it moves the experiment
    to the delete folder.
    
    Parameters
    ----------
    savedir : str
        Directory of the experiment
    backup_flag : bool, optional
        If true, instead of deleted is moved to delete folder, by default False
    """
    # get experiment id
    exp_id = os.path.split(savedir)[-1]
    assert(len(exp_id) == 32)

    # get paths
    savedir_base = os.path.dirname(savedir)
    savedir = os.path.join(savedir_base, exp_id)

    if backup_flag:
        # create 'deleted' folder 
        dst = os.path.join(savedir_base, 'deleted', exp_id)
        os.makedirs(dst, exist_ok=True)

        if os.path.exists(dst):
            shutil.rmtree(dst)
    
    if os.path.exists(savedir):
        if backup_flag:
            # moves folder to 'deleted'
            shutil.move(savedir, dst)
        else:
            # delete experiment folder 
            shutil.rmtree(savedir)

    # make sure the experiment doesn't exist anymore
    assert(not os.path.exists(savedir))

def saveJSON(fname, data, makedirs = True):
    """Save data into a json file.
    Parameters
    ----------
    fname : str
        Name of the json file
    data : [type]
        Data to save into the json file
    makedirs : bool, optional
        If enabled creates the folder for saving the file, by default True
    """
    dirname = os.path.dirname(fname)
    if makedirs and dirname != '':
        os.makedirs(dirname, exist_ok=True)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)

def loadJSON(fname, decode = None):  # TODO: decode???
    """Load a json file.
    Parameters
    ----------
    fname : str
        Name of the file
    decode : [type], optional
        [description], by default None
    Returns
    -------
    [type]
        Content of the file
    """
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d


def readText(fname):
    """Loads the content of a text file.
    Parameters
    ----------
    fname : str
        File name
    Returns
    -------
    list
        Content of the file. List containing the lines of the file
    """
    with open(fname, "r", encoding="utf-8", errors='replace') as f:
        lines = f.readlines()
    return lines


def loadPKL(fname):
    """Load the content of a pkl file.
    Parameters
    ----------
    fname : str
        File name
    Returns
    -------
    [type]
        Content of the file
    """
    with open(fname, "rb") as f:
        return pickle.load(f)


def savePKL(fname, data, with_rename = True, makedirs = True):
    """Save data in pkl format.
    Parameters
    ----------
    fname : str
        File name
    data : [type]
        Data to save in the file
    with_rename : bool, optional
        [description], by default True
    makedirs : bool, optional
        If enabled creates the folder for saving the file, by default True
    """
    # Create folder
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

def cartesian(exp_config, remove_none = False):
    """Cartesian experiment config.
    It converts the exp_config into a list of experiment configuration by doing
    the cartesian product of the different configuration. It is equivalent to
    do a grid search.
    Parameters
    ----------
    exp_config : str
        Dictionary with the experiment Configuration
    Returns
    -------
    exp_list: str
        A list of experiments, each defines a single set of hyper-parameters
    """
    exp_config_copy = copy.deepcopy(exp_config)

    # Make sure each value is a list
    for k, v in exp_config_copy.items():
        if not isinstance(exp_config_copy[k], list):
            exp_config_copy[k] = [v]

    # Create the cartesian product
    exp_list_raw = (dict(zip(exp_config_copy.keys(), values))
                    for values in itertools.product(*exp_config_copy.values()))

    # Convert into a list
    exp_list = []
    for exp_dict in exp_list_raw:
        # remove hparams with None
        if remove_none:
            to_remove = []
            for k, v in exp_dict.items():
                if v is None:
                    to_remove += [k]
            for k in to_remove:
                del exp_dict[k]
        exp_list += [exp_dict]

    return exp_list

def hashDict(exp_dict):
    """Create a hash for an experiment.
    Parameters
    ----------
    exp_dict : dict
        An experiment, which is a single set of hyper-parameters
    Returns
    -------
    hash_id: str
        A unique id defining the experiment
    """
    dict2hash = ""
    if not isinstance(exp_dict, dict):
        raise ValueError('exp_dict is not a dict')

    for k in sorted(exp_dict.keys()):
        if '.' in k:
            raise ValueError(". has special purpose")
        elif isinstance(exp_dict[k], dict):
            v = hashDict(exp_dict[k])
        elif isinstance(exp_dict[k], tuple):
            raise ValueError("tuples can't be hashed yet, consider converting tuples to lists")
        else:
            v = exp_dict[k]

        dict2hash += os.path.join(str(k), str(v))

    hash_id = hashlib.md5(dict2hash.encode()).hexdigest()

    return hash_id

def loadTorch(fname, map_location = None):
    """Load the content of a torch file.
    Parameters
    ----------
    fname : str
        File name
    map_location : [type], optional
        Maping the loaded model to a specific device (i.e., CPU or GPU), this
        is needed if trained in CPU and loaded in GPU and viceversa, by default
        None
    Returns
    -------
    [type]
        Loaded torch model
    """
    obj = torch.load(fname, map_location=map_location)

    return obj


def saveTorch(fname, obj):
    """Save data in torch format.
    Parameters
    ----------
    fname : str
        File name
    obj : [type]
        Data to save
    """
    # Create folder
    os.makedirs(os.path.dirname(fname), exist_ok=True)  # TODO: add makedirs parameter?

    # Define names of temporal files
    fname_tmp = fname + ".tmp"  # TODO: Make the safe flag?

    torch.save(obj, fname_tmp)
    if os.path.exists(fname):
        os.remove(fname)
    os.rename(fname_tmp, fname)