import torch
import torchvision.models as models
import imagenet_synsets
def get_prediction_ranks(scores):
    # Sort the scores in descending order and get the indices of the sorted scores
    sorted_indices = torch.argsort(scores, descending=True)

    # Get the ranks of the sorted scores by finding the index of each sorted score in the original scores
    ranks = torch.zeros_like(sorted_indices)
    classnames = [None for _ in (sorted_indices)]
    for i, index in enumerate(sorted_indices):
        ranks[index] = i
        # classname = imagenet_synsets.synsets[i]['label']
        classname = get_classname(i)
        classnames[index] = (classname)

    # Print the ranks of the predicted class scores
    print('Ranks of the predicted class scores:')
    for i in range(5):
        print('Rank {}: Class index = {}, Score = {:.4f}, Class name = {}'.format(i+1, sorted_indices[i], scores[sorted_indices[i]],classnames[i]))
    return ranks,classnames

def get_classname(i):
    return imagenet_synsets.synsets[i]['label']
    