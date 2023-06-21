import os
import glob
import imagenet_localization_parser
from synset_utils import get_synset_id,synset_id_to_imagenet_class_ix
IMAGENET_ROOT = "/root/bigfiles/dataset/imagenet"
def find_images_for_class_id(target_id):
    image_paths = sorted(glob.glob(os.path.join(IMAGENET_ROOT,'images','val','*.JPEG')))
    found_impaths = []
    for i,impath in enumerate(image_paths):

        imroot = os.path.basename(impath).split('.')[0]
        bbox_info = imagenet_localization_parser.get_voc_label(
            root_dir = os.path.join(IMAGENET_ROOT,'bboxes','val'),
            x = imroot)
    #     print(bbox_info)
        # import ipdb;ipdb.set_trace()
        synset_id = bbox_info['annotation']['object'][0]['name']
    #     print(synset_id)
        im_target_id = synset_id_to_imagenet_class_ix(synset_id)
        if im_target_id == target_id:
            found_impaths.append(impath)
    #     print(target_id)
        import imagenet_synsets
        classname = imagenet_synsets.synsets[im_target_id]['label']
        classname = classname.split(',')[0]
        if False:
            classname = '_'.join(classname)
        else:
            classname = classname.replace(' ','_')                
        # assert False, 'untested for imagenet'
        # import ipdb;ipdb.set_trace()
        target_ids = [im_target_id]
        classnames = [classname]
        # import ipdb;ipdb.set_trace()
        # bbox_info['annotation']['segmented']
        # bbox_info['annotation']['object'][0].keys()
        # import ipdb;ipdb.set_trace()
        # assert False
    return found_impaths

print(find_images_for_class_id(153))