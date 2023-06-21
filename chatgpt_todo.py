#================================================
# Invert
    #==================================================
    in pytorch given a mask tensor, and 2 image tensors, calculate the laplacian blending of the 2
    #==================================================
    in pytorch give me the code to:
    1. attach hooks to capture the features from layers for style transfer. set the default names of the layers to the ones typically used from vgg type networks
    2. calculate the style loss per layer, as well as the total style loss
    #==================================================
    in pytorch given 2 image tensors, and the number of linear interpolations required as k, calculate the linear interpolation between the 2. 
    #==================================================
    given an image tensor, calculate its style features and get the style loss from 2 reference style features. 
    #==================================================
    - in pytorch, given the features, write the code to dump it to a running pickle file which contains the features from previous iteration
    - write the code to read the pickle, and calculate the top 2 pca components from the features in the pickle file. use sklearn for this
    - project the features to the 2 top components
    #==================================================
    write the code to plot a trajectory consisting of x and y lists. mark the origin by 'o' the target by a filled o, and the trajectory is a solid line.the function should be called plot_trajectory, it accepts x, y the title and the savename. save the plot under savename. do not show the plot, only draw it, and then close the figure after saving
#==================================================
# GPNN road
    #==================================================
    # to see which order extract_patches flattens patches
    in pytorch given a size H,W, create a tensor of coordinates of size H,W,2 where the first channel contains the y coordinate and the second the x coordinate
    #==================================================
    # convert I to y,x indexes
    in pytorch convert a linear index to the y,x coordinate. assume the flattening was done by arranging rows after one another
    #==================================================
    # flow metric
    given 2 tensors locs1 and locs2, which contain yx coordinates, and a max_window_size, calculate the elementwise displacement between locs1 and locs2, normalized by the maximum displacement of max_window_size * sqrt(2)
    #==================================================
    # upsizing the low dimensional flow
    in pytorch given the flow at a scale, which is of size (2,H0,W0) where the 2 channels are the y and x coordinate. upsample this to a size (2,H1,W1).calculate the diplacement map using the function defined in the previous step
    #==================================================
    # extracting regions from images
    in pytorch you are given a list of coordinates where each coordinate is a 3 tuple of (b,y,x) where b is the image id, y and x are the coordinates of a center pixel in it. you are also given a window_size. for each image referred to in the list, extract a indow_size times window_size region around the center pixel. return all the extracted regions as a list.
#==================================================
# edge map
    #==================================================
    in pytorch given the magnitude and angle of the edge, compute a 3x3 window which contains the continuation of the edge with the same magnitude and angle
    #==================================================
    TODO: in pytorch define a gradient descent based poisson hole filling scheme . edges of the hole are consistent with those of the boundary. i am optimizing for the values to fill the hole.
    #==================================================
    in pytorch give me the code to make a gaussian window of patch_size times patch_size, with a given sigma