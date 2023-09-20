import numpy as np
import torch
import skimage.io
import os
from matplotlib import pyplot as plt
import colorful
import argparse
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
try:
    import dutils
except ModuleNotFoundError as e:
    dutils = argparse.Namespace()
    # dutils.get = vars(dutils).get
    pass
from argparse import Namespace
import ipdb
import threading
import os
import seaborn as sns
mu = [0.485, 0.456, 0.406]
sigma = [0.229, 0.224, 0.225]
DEBUG_DIR = 'debugging'
SAVE_DIR = DEBUG_DIR
SYNC = False
SYNC_DIR = SAVE_DIR
REMOTE_SYNC_PARENT = "fong-invert"
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)
tensor_to_numpy = lambda t:t.detach().cpu().numpy()

def img_save(img, savename):
    print(colorful.tan(f"saving {os.path.abspath(savename)}"))
    '''
    An adaptive image saving method, that works for numpy arrays, torch tensors. Also can accomodate multiple shapes of the input as well as ranges of values    
    '''    
    if isinstance(img,torch.Tensor):
        img = tensor_to_numpy(img)
    
    # shape:
    print('img has shape: ',img.shape)
    if img.ndim == 4:
        print('got 4d input, assuming first channel is batch, saving the fist image')
        img = img[0]
    if img.ndim == 3:
        if img.shape[0] == 1:
            print('got input with 1 channel, assuming grayscale')
            img = img[0]
        elif img.shape[0] == 3:
            print('got input with 3 channels')
            img = np.transpose(img,(1,2,0))
    if img.min() >= 0 and img.max() <= 1:
        print('got img with values in [0,1] range')
    else:
        print('TODO: figure out what to do with min < 0 and max> 1')
    skimage.io.imsave(savename,img)

def sync_to_gdrive(foldername):
    folderbasename = os.path.basename(foldername.rstrip(os.path.sep))
    # oipdb('sync-gdrive')
    # import ipdb; ipdb.set_trace()
    print(f'Syncing {foldername} to {REMOTE_SYNC_PARENT}/{folderbasename}')
    os.system(f'rclone sync -Pv {foldername} aniketsinghresearch-gdrive:{REMOTE_SYNC_PARENT}/{folderbasename}')
    
def img_save2(img,basename,syncable=False):
    # prefix = dutils.get('save_prefix','')
    prefix = ''
    d = SAVE_DIR
    if prefix:
        d = os.path.join(d,prefix)
    if not os.path.exists(d):
        os.makedirs(d)        
    savename= os.path.join(d,basename)
    img_save(img,savename)
    # import ipdb; ipdb.set_trace()
    if SYNC:
        if syncable:
            sync_to_gdrive(SYNC_DIR)    
    return savename

def save_plot(y,title,savename,x=None):
    print(colorful.tan(f"saving {os.path.realpath(savename)}"))
    plt.figure()
    if x is None:
        plt.plot(y)
    else:
        plt.plot(x,y)
    plt.title(title)
    plt.show()
    plt.savefig(savename)
    plt.close()
    
def save_plot2(y,title,basename,x=None,syncable=False):
    # prefix = dutils.get('save_prefix','')
    prefix = ''
    d = SAVE_DIR
    # import ipdb; ipdb.set_trace()
    if prefix:
        d = os.path.join(d,prefix)
    if not os.path.exists(d):
        os.makedirs(d)        
    savename = os.path.join(d,basename)
    save_plot(y,title,savename,x=x)
    if SYNC:
        if syncable:
            sync_to_gdrive(SYNC_DIR)
    return savename

def save_histogram(values,title,savename):
    plt.figure()
    ax = plt.gca()
    # if x is None:
    #     plt.plot(y)
    # else:
    #     plt.plot(x,y)
    sns.kdeplot(values,ax= ax)
    plt.title(title)
    plt.show()
    plt.savefig(savename)
    plt.close()
    
    pass

def save_histogram2(values,title,basename,syncable=False):
    # prefix = dutils.get('save_prefix','')
    prefix = ''
    d = SAVE_DIR
    # import ipdb; ipdb.set_trace()
    if prefix:
        d = os.path.join(d,prefix)
    if not os.path.exists(d):
        os.makedirs(d)        
    savename = os.path.join(d,basename)
    save_histogram(values,title,os.path.join(d,basename))
    if SYNC:
        if syncable:
            sync_to_gdrive(SAVE_DIR)
    return savename

class MyNamespace(Namespace):
    def __init__(self):
        super().__init__()
        self._attr = []
        self.LOCKED = []  # Initialize the list of locked attributes
        # self._values = {}  # Initialize a dictionary to hold attribute values
        # import ipdb; ipdb.set_trace()
        # debug as_dict
        self.as_dict()
    def set_and_lock(self,name,value):
        self.unlock(name)
        self.__dict__[name] = value
        self.__dict__['_attr'] += [name]
        self.__dict__['_attr'] = list(set(self._attr))
        self.lock(name)        
    def lock(self,name):
        if name not in self.LOCKED:
            self.LOCKED.append(name)
    def unlock(self,name):
        if name in self.LOCKED:
            ix = self.LOCKED.index(name)
            del self.LOCKED[ix]
        # import ipdb; ipdb.set_trace()
    def get(self,name,default=None):
        return self.__dict__.get(name,default)
    def reset(self):
        self.LOCKED = []
    def __setattr__(self, name, value):
        if name not in self.__dict__:
            if name in ['_attr','LOCKED']:
                self.__dict__[name] = value
                return
            self.__dict__[name] = value
            self.__dict__['_attr'] += [name]
            self.__dict__['_attr'] = list(set(self._attr))
        if name in self.LOCKED:
            # raise AttributeError(f"Attribute {name} is locked and cannot be set.")
            print(colorful.red(f"Attribute {name} is locked and cannot be set. using value {self.__dict__[name]}"))
            return 
        self.__dict__[name] = value
    def as_dict(self):
        out = {}
        for name in self._attr:
            if name in self.__dict__:
                out[name]  = self.__dict__[name]
        return out

def set_save_dir(*parts,purge=False):
    globals()['SAVE_DIR'] = os.path.join(*parts)
    if os.path.exists(globals()['SAVE_DIR']):
        if purge:
            os.system(f"rm -rf {globals()['SAVE_DIR']}")
    if not os.path.isdir((globals()['SAVE_DIR'])):
        os.makedirs(globals()['SAVE_DIR'])
    


def cipdb(flag,val='1'):
    if os.environ.get(flag,False) == val:
        import ipdb; ipdb.set_trace()
    pass
oipdb_points = {}
def oipdb(name,count=1):
    if name not in oipdb_points:
        oipdb_points[name] = 1
        if oipdb_points[name] <= count:
            import ipdb; ipdb.set_trace()
    else:
        oipdb_points[name] += 1


def run_in_another_thread(f,args=[]):
    save_thread = threading.Thread(target=f, args=args)
    save_thread.start()
    
    
import torch
import gc
def print_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
                # print(gc.get_referrers(obj))
        except:
            pass
def printl(*messages,log_file='log.txt'):
    d =SAVE_DIR
    print(*messages)
    with open(os.path.join(d,log_file),'a') as f:
        message_str = ' '.join([str(m) for m in messages])
        f.write(message_str+'\n')

import builtins
import importlib
def import_and_reload(modulename):
    mod = builtins.__import__(modulename)
    importlib.reload(mod)
    return mod

#===========================================================
import re

def is_magic(name):
    """Return True if the name is a magic method or variable."""
    return re.match(r'^__(\w+)__$', name) is not None

def list_variables():
    """List all items in locals() and globals() that are not magic methods or variables."""
    all_vars = set(locals().keys()).union(globals().keys())
    for var_name in sorted(all_vars):
        if not is_magic(var_name):
            print(var_name)

import copy
# list_variables()
class TrackChange():
    def __init__(self,init_value,name=None):
        self.name = name
        if isinstance(init_value,torch.Tensor):
            init_value = tensor_to_numpy(init_value)
        self.value = copy.deepcopy(init_value)
        self.change = []
        pass
    def update(self,new_value):
        if isinstance(new_value,torch.Tensor):
            new_value = tensor_to_numpy(new_value)        
        change = np.abs(new_value - self.value).sum()
        self.change.append(change)
        self.value = copy.deepcopy(new_value)
        pass
    pass