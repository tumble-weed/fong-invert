import sys
sys.path.append('/root/evaluate-saliency-4/fong-invert/SCM')
import os
# os.chdir('/root/evaluate-saliency-4/fong-invert/SCM')
from lib.models import *
import torch
import numpy as np
import cv2    
# from config.default import config as cfg
from timm.models import create_model as create_deit_model
# args = update_config()
from PIL import Image
import torchvision.transforms as transforms
import skimage.io

tensor_to_numpy = lambda t:t.detach().cpu().numpy()

#----------------------------------------------------------------
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
global model

def load_model(
    checkpoint_path = "/root/evaluate-saliency-4/fong-invert/SCM/Imagenet_SCM/ckpt/model_best.pth",
    NUM_CLASSES = 1000,
    # ARCH = "deit_tscam_small_patch16_224",
    ARCH = "deit_scm_small_patch16_224", 
    ):
    global model
    device = 'cuda'
    if 'model' not in globals():
        model = create_deit_model(
            ARCH,
            pretrained=True,
            num_classes=NUM_CLASSES,
            drop_rate=0.0,
            drop_path_rate=0.1,
            drop_block_rate=None,
        )
        model.to(device)
        checkpoint = torch.load(checkpoint_path)
        pretrained_dict = {}

        for k, v in checkpoint['state_dict'].items():
            k_ = '.'.join(k.split('.')[1:])

            pretrained_dict.update({k_: v})    
        model.load_state_dict(pretrained_dict)
        model.eval()


mu = [0.485, 0.456, 0.406]
sigma = [0.229, 0.224, 0.225]    
size = (224,224)
transform = transforms.Compose([
    transforms.Resize(size=size),
    transforms.CenterCrop(size=size),
    transforms.ToTensor(),
    transforms.Normalize(mu, sigma),
])    
def read_img(impath = "/root/evaluate-saliency-4/fong-invert/grace_hopper.jpg"):
    ref = transform(Image.open(impath)).unsqueeze(0)
    return ref

def resizeNorm(cam, size=(224, 224)):
    cam = cv2.resize(cam, (size[0], size[1]))
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min)
    return cam    



def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation
    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list

def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                        .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                        "scoremap.min()={}, scoremap.max()={}."
                        .format(scoremap.min(), scoremap.max()))    
        
def draw_bbox(image, iou, gt_box, pred_box, draw_box=True, draw_txt=True):

    def draw_bbox(img, box1, box2, color1=(0, 0, 255), color2=(0, 255, 0)):
        for i in range(len(box1)):
            # cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.rectangle(img, (box1[i,0], box1[i,1]), (box1[i,2], box1[i,3]), color1, 1)
        for i in range(len(box2)):
            cv2.rectangle(img, (box2[i,0], box2[i,1]), (box2[i,2], box2[i,3]), color2, 1)
        return img

    def mark_target(img, text='target', pos=(25, 25), size=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), size)
        return img

    boxed_image = image.copy()
    if draw_box:
        # draw bbox on image
        boxed_image = draw_bbox(boxed_image, gt_box, pred_box)
    if draw_txt:
        # mark the iou
        mark_target(boxed_image, '%.1f' % (iou * 100), (140, 30), 2)

    return boxed_image
def get_localization(model,ref):
    #=========================================================================
    cls_scores, cams, _, _ = model(ref)
    cls_scores = cls_scores.cpu().tolist()
    cams = cams.cpu().tolist()    
    max_k = 1

    cls_scores = np.array(cls_scores)
    topk_ind = torch.topk(torch.from_numpy(
    cls_scores), max_k)[-1].numpy()  # [B K]    
    cams = np.array([np.take(a, idx, axis=0) for (a, idx) in zip(
                cams, topk_ind)])  # index for each batch # [B topk H W]    

    scoremap = cams[0,0]
    RESHAPE_SIZE=224
    _CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
    multi_contour_eval=False
    scoremap = resizeNorm(scoremap, (RESHAPE_SIZE, RESHAPE_SIZE))
    opt_thred = 0.5
    boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
                    scoremap=scoremap,
                    scoremap_threshold_list=[opt_thred],
                    multi_contour_eval=multi_contour_eval)    
    ref_im = tensor_to_numpy(ref.permute(0,2,3,1))
    boxed_image = draw_bbox(ref_im[0], opt_thred, boxes_at_thresholds[0], boxes_at_thresholds[0], draw_box=True, draw_txt=True)
    return cls_scores,topk_ind,boxed_image,boxes_at_thresholds

def get_wsol(model,ref,target_class=None):
    RESHAPE_SIZE=224
    CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
    multi_contour_eval=False
    max_k = 1
    opt_thred = 0.5
    #==============================================
    cls_scores, cams, _, _ = model(ref)
    cls_scores = cls_scores.cpu().tolist()
    cams = cams.cpu().tolist()    
    cls_scores = np.array(cls_scores)
    if target_class is None:
        topk_ind = torch.topk(torch.from_numpy(
        cls_scores), max_k)[-1].numpy()  # [B K]    
    else:
        topk_ind = [target_class]
    cams = np.array([np.take(a, idx, axis=0) for (a, idx) in zip(
                cams, topk_ind)])  # index for each batch # [B topk H W]        
    scoremap = cams[0,0]
    #==============================================
    scoremap = resizeNorm(scoremap, (RESHAPE_SIZE, RESHAPE_SIZE))
    boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
                    scoremap=scoremap,
                    scoremap_threshold_list=[opt_thred],
                    multi_contour_eval=multi_contour_eval)    
    
    ref_im = tensor_to_numpy(ref.permute(0,2,3,1))
    boxed_image = draw_bbox(ref_im[0], opt_thred, boxes_at_thresholds[0], boxes_at_thresholds[0], draw_box=True, draw_txt=True)
    return boxes_at_thresholds,number_of_box_list,scoremap,boxed_image
def test():
    load_model()
    global model
    impath = "/root/evaluate-saliency-4/fong-invert/grace_hopper.jpg"
    # from PIL import Image
    # import torchvision.transforms as transforms
    # mu = [0.485, 0.456, 0.406]
    # sigma = [0.229, 0.224, 0.225]    
    # size = (224,224)
    # transform = transforms.Compose([
    #     transforms.Resize(size=size),
    #     transforms.CenterCrop(size=size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mu, sigma),
    # ])    
    # ref = transform(Image.open(impath)).unsqueeze(0)
    ref = read_img(impath = "/root/evaluate-saliency-4/fong-invert/grace_hopper.jpg")
    boxes_at_thresholds,number_of_box_list,scoremap,boxed_image = get_wsol(model,ref,target_class=None)
    #=========================================================================    
    import ipdb; ipdb.set_trace()
#===============================================================
import glob
import os
import pickle
import torch
def run_on_inversion_results(pattern = "debugging/invert_noisy/run_multi_class_best_alexnet/*_noisy_*/"):
    load_model()
    global model
    device = 'cuda'
    
    imdirs = glob.glob( pattern)
    print(len(imdirs))
    
    for imdir in imdirs:
        pklname = glob.glob(os.path.join(imdir,'*.pkl'))
        assert len(pklname) == 1
        pklname = list(pklname)[0]
        with open(pklname,'rb') as f:
            trends = pickle.load(f)
            x__ = pickle.load(f)
            opts_dict = pickle.load(f)
        if 'boxes_at_thresholds' not in trends:
            x__ = torch.tensor(x__,device='cuda').float()
            x_ = (x__ - 0.5)*2*3
            x_ = torch.nn.functional.interpolate(x_,size=(224,224),mode='bilinear',align_corners=False)
            boxes_at_thresholds,number_of_box_list,scoremap,boxed_image = get_wsol(model,x_,target_class=opts_dict['target_class'][0])
            bbox_areas = []
            for boxes_x0y0x1y1 in boxes_at_thresholds:
                assert boxes_x0y0x1y1.shape[0] == 1
                x0,y0,x1,y1 = boxes_x0y0x1y1[0]
                area = (y1-y0 + 1) * (x1-x0 + 1)
                bbox_areas.append(area)
            trends['boxes_at_thresholds'] = boxes_at_thresholds
            trends['number_of_box_list'] = number_of_box_list
            trends['scoremap'] = scoremap
            trends['boxed_image'] = boxed_image
            trends['bbox_areas'] = bbox_areas
        
            with open(pklname,'wb') as f:
                pickle.dump(trends,f)
                pickle.dump(x__,f)
                pickle.dump(opts_dict,f)
            assert False
        else:
            for k in ['boxes_at_thresholds',
                      'number_of_box_list',
                      'scoremap',
                      'boxed_image',
                      'bbox_areas'
                      ]:
                assert k in trends
            print('found keys')
#===============================================================
if __name__ == '__main__':
    # test()
    for modelname in ["alexnet","vgg"]:
        for noise_type in ["noisy","noiseless"]:
            run_on_inversion_results(pattern = f"debugging/invert_noisy/run_multi_class_best_{modelname}/*_{noise_type}_*/")
    