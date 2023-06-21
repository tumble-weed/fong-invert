def get(key,default,d=locals()):
    return d.get(key,default)

def getif(key,default_if_flag,default_if_not_flag,flag,d=locals()):
    if d.get(flag,False):
        return d.get(key,default_if_flag)
    return default_if_not_flag
    