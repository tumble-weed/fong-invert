import torch
asset = {}
def grad_alignment(g,name='unnamed',asset=asset):
    if name not in asset:
        asset[name] = (g).detach().clone()
        return 0
    prev_g = asset[name]
    cosine_sim = torch.nn.functional.cosine_similarity(g.view(-1),prev_g.view(-1),dim=0)
    asset[name] = (g).detach().clone()
    return cosine_sim.item()