import torch
def get_d_loss_by_dt(x,sharpness,t,A):
    s_k = torch.sigmoid(sharpness*(x - t))
    area =torch.sum(s_k**2)
    d_loss_by_dt = 2*(area - A) * (2 * (s_k**2) * (1 - s_k**2)* (-sharpness) ).sum()
    return d_loss_by_dt
if False and "what was this?":    
    nx = 2
    sharpness = 1
    tcheck = 0.1
    x = torch.randn(nx)
    A = (torch.sigmoid(sharpness*(x - tcheck))**2).sum().detach()
    d_loss_by_dt = get_d_loss_by_dt(x,sharpness,tcheck,A)
    del_t = 0.001
    A_near = (torch.sigmoid(sharpness*(x - (tcheck + del_t)))**2).sum().detach()
    manual_d_loss_by_dt = (A_near - A)/del_t
    # predicted_change_in_A = d_loss_by_dt * del_t
    # print(change_in_A,predicted_change_in_A)
    print(manual_d_loss_by_dt,d_loss_by_dt)
def get_d_t_by_d_x(x,sharpness,t):
    s_k = torch.sigmoid(sharpness*(x - t))
    # d_t_by_d_x = (2 * (s_k**2) * (1 - s_k**2)* (-sharpness) ).sum()
    Nr = (s_k**2) * (1 - s_k)
    Dr = ((s_k**2) * (1 - s_k)).sum()
    d_t_by_d_x = Nr/Dr
    return d_t_by_d_x
if False and "check with small deviation":    
    nx = 2
    sharpness = 0.1
    tcheck = 0.1
    x = torch.randn(nx)
    A = (torch.sigmoid(sharpness*(x - tcheck))**2).sum().detach()
    d_t_by_d_x = get_d_t_by_d_x(x,sharpness,tcheck)
    del_x = 0.1 * 2*(torch.rand(nx) - 0.5)
    x_near = x.detach().clone() 
    x_near = x_near+ del_x
    t_near = tcheck + (d_t_by_d_x * del_x).sum()
    A_near = (torch.sigmoid(sharpness*(x_near - t_near))**2).sum().detach()
    A_wrong = (torch.sigmoid(sharpness*(x_near - tcheck))**2).sum().detach()
    print(f'A:{A:.6f},A_near:{A_near:.6f},A_wrong:{A_wrong:.6f}' )
    print(f'tcheck:{tcheck},t_near:{t_near}')
if True and "check with interpolation":
    nx = 200
    sharpness = 0.1
    t0 = 0.1
    max_xi_change = 10
    std_change = 100
    nsteps = 100
    x = torch.randn(nx)
    x_change =  std_change * torch.randn(nx).clip(-max_xi_change,max_xi_change)
    A0 = (torch.sigmoid(sharpness*(x - t0))**2).sum().detach()
    tnow = t0
    xnow = x.detach().clone()
    As = []
    del_x = x_change/nsteps
    for i in range(nsteps):
        d_t_by_d_x = get_d_t_by_d_x(xnow,sharpness,tnow)    
        # xold = xnow
        xnow = xnow+ del_x
        tnow = tnow + (d_t_by_d_x * del_x).sum()
        Anow = (torch.sigmoid(sharpness*(xnow - tnow))**2).sum().detach()
        As.append(Anow.item())
    """
    print('starting and ending area')
    print(A0.item(),As[-1].item())
    print('using the original t')
    """
    A_wrong = (torch.sigmoid(sharpness*(xnow - t0))**2).sum().detach()
    print(f'A:{A0.item():.6f},Atrack:{As[-1]:.6f},A_wrong:{A_wrong.item():.6f}' )
    print(f't0:{t0},t_near:{tnow}')    