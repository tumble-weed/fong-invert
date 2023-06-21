from scipy import optimize
from scipy.interpolate import interp1d
import torch
import os
import colorful
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
import numpy as np
# mask_ = (normalized_ranks).detach().cpu().numpy().flatten()
#import pdb;pdb.set_trace()
# mask_ = normalized_ranks.flatten()

#import pdb;pdb.set_trace()
def get_thresholds(mask,sharpness = 20,required_areas=None,l=1,mode="cumsum"):
    device= mask.device
    mask_ = mask.flatten()
    K =1
    MAX_AREA = 1
    def get_area(t,mask_=mask_,s=sharpness,jac=False,l=l):
        if sharpness < 100:
            assert mask_.ndim == 1
            # assert t.ndim == 1
            if t.ndim == 0:
                t = np.atleast_1d(t)
            t= torch.tensor(t).to(mask_.device)
            # return sigmoid(s*(mask_[None,:] - t[:,None])).sum(axis=-1)
            S = torch.sigmoid(s*(mask_[None,:] - t[:,None]))
            if l ==1:
                area = S.sum(axis=-1)/MAX_AREA
            elif l ==2:
                area = (S**2).sum(axis=-1)/MAX_AREA
            return area
        else:
            # assert False
            t= torch.tensor(t).to(mask_.device)
            return (mask_[None,:] >= t[:,None]).float().sum(axis=-1)    
    

    METHOD = "root"
    if METHOD == "root":
        traj_d,traj_t,traj_a= [],[],[]
        
        def area_discrepancy(tpre,A=None,s=sharpness):
            A = A/MAX_AREA
            # a = get_area(t,jac=False)
            t = tpre
            a= get_area(t)
            # return np.abs((a - A))
            # jac = np.clamp(jac,1e-8)
            traj_d.append(tensor_to_numpy(a - A))
            traj_t.append((t))
            traj_a.append((a))
            # print(t,(a - A))
            return tensor_to_numpy((a - A))
        if True:
            all_sols= []
            all_errs = []
            for A in required_areas:
                sol = optimize.root(area_discrepancy, 
                                # np.linspace(mask_.min().item(),mask_.max().item(),len(required_areas)), 
                                x0=0.0,args=(A,),method='lm') #'hybr' didnt work well
                all_sols.append(sol.x)
                all_errs.append(area_discrepancy(np.atleast_1d(sol.x),A=A))
            all_sols = np.array(all_sols).squeeze()
            all_errs = np.array(all_errs).squeeze()
    os.environ['IGNORE_AREA_ERROR'] = '1'
    print(

        colorful.yellow_on_red(
            'setting IGNORE_AREA_ERROR:' + os.environ['IGNORE_AREA_ERROR']
        )
    )
    if (sum([err for err in all_errs]) < 1e-1) or os.environ.get('IGNORE_AREA_ERROR',False) == '1':
        # rel_rank_thresh = [sol for sol in all_sols]
        rel_rank_thresh = all_sols
        if (required_areas[1:] > required_areas[:-1]).all():
            assert ((rel_rank_thresh[1:] - rel_rank_thresh[:-1]) <= 0).all()
        else:
            assert False, "was hoping for later areas to be larger"                        
    else:
        # raise Exception        
        import ipdb; ipdb.set_trace()

    rel_rank_thresh = torch.tensor(rel_rank_thresh).float().to(device)
    return rel_rank_thresh
    
"""

def get_thresholds2(mask,sharpness = 20,required_areas=None,l=1,mode="cumsum"):
    device= mask.device
    mask_ = mask.flatten()
    K =1
    MAX_AREA = 1
    def get_area(t,mask_=mask_,s=sharpness,jac=False,l=l):
        if sharpness < 100:
            assert mask_.ndim == 1
            assert t.ndim == 1
            t= torch.tensor(t).to(mask_.device)
            # return sigmoid(s*(mask_[None,:] - t[:,None])).sum(axis=-1)
            S = torch.sigmoid(s*(mask_[None,:] - t[:,None]))
            if l ==1:
                area = S.sum(axis=-1)/MAX_AREA
            elif l ==2:
                area = (S**2).sum(axis=-1)/MAX_AREA
            return area
        else:
            assert False
            t= torch.tensor(t).to(mask_.device)
            return (mask_[None,:] >= t[:,None]).float().sum(axis=-1)    
    

    METHOD = "fsolve"
    if METHOD == "fsolve":
        traj_d,traj_t,traj_a= [],[],[]
        
        def area_discrepancy(tpre,A=None,s=sharpness):
            A = A/MAX_AREA
            # a = get_area(t,jac=False)
            t = tpre
            a= get_area(t)
            # return np.abs((a - A))
            # jac = np.clamp(jac,1e-8)
            traj_d.append(tensor_to_numpy(a - A))
            traj_t.append((t))
            traj_a.append((a))
            # print(t,(a - A))
            return tensor_to_numpy((a - A))
        if True:
            all_sols= []
            all_errs = []
            for A in required_areas:
                sol = optimize.fsolve(area_discrepancy, 
                                # np.linspace(mask_.min().item(),mask_.max().item(),len(required_areas)), 
                                x0=0.0,args=(A,)) #'hybr' didnt work well
                assert sol.shape == (1,)
                sol = sol[0]
                all_sols.append(sol)
                all_errs.append(area_discrepancy(np.atleast_1d(sol),A=A))
            all_sols = np.array(all_sols)
            all_errs = np.array(all_errs)
    if sum([err for err in all_errs]) < 1e-1:
        # rel_rank_thresh = [sol for sol in all_sols]
        rel_rank_thresh = all_sols
        if (required_areas[1:] > required_areas[:-1]).all():
            assert ((rel_rank_thresh[1:] - rel_rank_thresh[:-1]) <= 0).all()
        else:
            assert False, "was hoping for later areas to be larger"                        
    else:
        # raise Exception        
        import ipdb; ipdb.set_trace()

    rel_rank_thresh = torch.tensor(rel_rank_thresh).float().to(device)
    return rel_rank_thresh
    
    
def get_thresholds1(mask,sharpness = 20,required_areas=None,l=1,mode="cumsum"):
    device= mask.device
    mask_ = mask.flatten()
    K =1
    MAX_AREA = 1
    def get_area(t,mask_=mask_,s=sharpness,jac=False,l=l):
        if sharpness < 100:
            assert mask_.ndim == 1
            assert t.ndim == 1
            t= torch.tensor(t).to(mask_.device)
            # return sigmoid(s*(mask_[None,:] - t[:,None])).sum(axis=-1)
            S = torch.sigmoid(s*(mask_[None,:] - t[:,None]))
            if l ==1:
                area = S.sum(axis=-1)/MAX_AREA
            elif l ==2:
                area = (S**2).sum(axis=-1)/MAX_AREA
            if jac:
                if l ==1:
                    jac =  (-s * S * (1 - S)).sum(axis=-1)/MAX_AREA
                elif l == 2:
                    jac =  ( (2* S) * -s * S * (1 - S)).sum(axis=-1)/MAX_AREA
                # import ipdb; ipdb.set_trace()
                return area,jac
            return area
        else:
            t= torch.tensor(t).to(mask_.device)
            return (mask_[None,:] >= t[:,None]).float().sum(axis=-1)    
    
    if True and 'exact areas':
        METHOD = "root"
        if METHOD == "root":
            traj_d,traj_t,traj_a= [],[],[]
            
            def area_discrepancy(tpre,A=required_areas,s=sharpness,mode=mode):
                A = A/MAX_AREA
                if mode == 't':
                    t = tpre
                elif mode == 'cumsum':
                    t = tpre.cumsum()
                d = t.shape[0]    
                # a = get_area(t,jac=False)
                a,jac_ = get_area(t,jac=True)
                if mode == 't':
                    jac = np.diag(tensor_to_numpy(jac_))
                elif mode == 'cumsum':
                    jac = np.tril(np.ones((d,d)))
                    jac = tensor_to_numpy(jac_[:,None]) * jac
                    # import ipdb; ipdb.set_trace()
                assert a.shape == t.shape
                # return np.abs((a - A))
                # jac = np.clamp(jac,1e-8)
                traj_d.append(tensor_to_numpy(a - A))
                traj_t.append((t))
                traj_a.append((a))
                # print(t,(a - A))

                return tensor_to_numpy((a - A)),jac
            if True:
                sol = optimize.root(area_discrepancy, 
                                # np.linspace(mask_.min().item(),mask_.max().item(),len(required_areas)), 
                                np.zeros((len(required_areas),)),
                                jac=True, method='lm') #'hybr' didnt work well
            print('Max Area Discrepancy:',area_discrepancy(sol.x)[0].max())
        elif METHOD == "minimize":
            traj_d,traj_t,traj_a= [],[],[]
            def area_discrepancy(tpre,A=required_areas,s=sharpness,mode=mode):
                if mode == 't':
                    t = tpre
                elif mode == 'cumsum':
                    t = tpre.cumsum()
                d = t.shape[0]    
                # a = get_area(t,jac=False)
                a,jac_ = get_area(t,jac=True)
                if mode == 't':
                    jac = np.diag(tensor_to_numpy(jac_))
                elif mode == 'cumsum':
                    jac = np.tril(np.ones((d,d)))
                    jac = tensor_to_numpy(jac_[:,None]) * jac
                    # import ipdb; ipdb.set_trace()
                assert a.shape == t.shape
                # return np.abs((a - A))
                # jac = np.clamp(jac,1e-8)
                traj_d.append(tensor_to_numpy(a - A))
                traj_t.append((t))
                traj_a.append((a))
                # print(t,(a - A))

                return tensor_to_numpy((a - A)**2).sum()#,jac
            if True:
                sol = optimize.minimize(area_discrepancy, 
                                # np.linspace(mask_.min().item(),mask_.max().item(),len(required_areas)), 
                                np.zeros((len(required_areas),)),
                                jac=False, method='Nelder-Mead') #'hybr' didnt work well            

        
        # import pdb;pdb.set_trace()
        # if sol.success:
        if (np.abs(sol.fun).sum() < 1e-1):
            if mode  == 'cumsum':
                rel_rank_thresh = sol.x.cumsum()
            else:
                rel_rank_thresh = sol.x
            if (required_areas[1:] > required_areas[:-1]).all():
                assert ((rel_rank_thresh[1:] - rel_rank_thresh[:-1]) <= 0).all()
            else:
                assert False, "was hoping for later areas to be larger"                        
        else:
            assert mode == 'cumsum'        
            reinit_t = np.zeros(sol.x.shape)
            reinit_t[1:] = sol.x[1:] * (sol.x[1:] < 0).astype(np.float32)
            sol2 = optimize.root(area_discrepancy, 
                                        # np.linspace(mask_.min().item(),mask_.max().item(),len(required_areas)), 
                                        reinit_t,
                                        jac=True, method='lm') #'hybr' didnt work well
            assert (np.abs(sol2.fun).sum() < 1e-1) 
            rel_rank_thresh = sol2.x.cumsum()
            # import ipdb; ipdb.set_trace()
            # raise Exception
        if False and 'random':
            print(colorful.red("returning random thresholds"))
            a = tensor_to_numpy(required_areas/required_areas[-1])
            F = interp1d(a,rel_rank_thresh,axis=0)
            steps = np.random.rand(len(required_areas))
            rel_ticks = steps.cumsum(axis=0)/steps.sum()
            rel_ticks = np.clip(rel_ticks,a.min(),a.max())
            rel_rank_thresh = F(rel_ticks)
        
        # rel_rank_thresh = np.linspace(rel_rank_thresh[0],rel_rank_thresh[1],rel_rank_thresh.shape[0]); print('setting rel_rank_thresh to just linspace')
        rel_rank_thresh = torch.tensor(rel_rank_thresh).float().to(device)
        return rel_rank_thresh
    elif False  and 'random':
        A = torch.cat(
            [required_areas[:1],required_areas[-1:]],
            dim = 0
        )
        
        def area_discrepancy(t,
            A=A):
            a = get_area(t)
            assert a.shape == t.shape
            # return np.abs((a - A))
            return tensor_to_numpy((a - A))
        sol = optimize.root(area_discrepancy, [mask_.min().item(),mask_.max().item()], jac=False, method='hybr')
        lims = sol.x            
        steps = torch.rand(len(required_areas)).to(device)
        rel_ticks = steps.cumsum(dim=0)/steps.sum()
        assert lims[0] > lims[1],'smaller area should have higher threshold'
        rel_rank_thresh = lims[0] + (lims[1] - lims[0])*rel_ticks
        # import pdb;pdb.set_trace()
        # rel_rank_thresh = rel_rank_thresh[::-1]
        # rel_rank_thresh = torch.sort(rel_rank_thresh)[0]
"""