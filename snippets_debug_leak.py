if False:
    if True:
        # _ =model(ref)
        activations = []
        model2 = torchvision.models.alexnet()
        model2.to('cuda')
        activations2 = setup_network(model2,layer,activations)
        with torch.no_grad():
            _ = model2(ref)
        with torch.no_grad():
            _ = model2(ref)
        with torch.no_grad():
            _ = model2(ref)                
    if True:
        hooks.hook_dict
        hooks.assets

    if True:
        del model2
        torch.cuda.empty_cache()
    if True:    
        del activations2[0]
        torch.cuda.empty_cache()
    if True:    
        del hooks.assets['forward',layer][0]
        torch.cuda.empty_cache()
    if True:    
        del hooks.activations[0]
        torch.cuda.empty_cache()
        
if False:
    if False:
        with torch.no_grad():
            _ = model(ref)
        with torch.no_grad():
            _ = model(ref)        
    _ = model(ref)   
    if True:
        del activations[0]
        del activations_gradcam[0]

        
        
        for outers in zip(activations_sparsity,noisy_activations_sparsity,avg_angles,avg_mags):
            for io,outer in enumerate(outers):
                print(io)
                try:
                    del outer[0]
                except IndexError:
                    pass
    if True:
        torch.cuda.empty_cache()
        
        del grads_gradcam[0]    
        torch.cuda.empty_cache()    
        
        
if True:
    _ = model(ref)   
    _.sum().backward()
    if True:
        activations_gradcam[0].grad = None
        torch.cuda.empty_cache()
        del activations_gradcam[0]
        torch.cuda.empty_cache()
