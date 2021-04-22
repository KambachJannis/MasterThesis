import math
import os
import cv2
import imgaug as ia
import matplotlib.pyplot as plt
import numpy as np
import torch
from imgaug import augmenters as iaa
from scipy import sparse
from scipy.interpolate import interp1d
from sklearn.metrics import pairwise_distances_argmin_min
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from imgaug.augmenters import Augmenter
import numpy as np
from imgaug import augmenters as iaa


class PASCALDataSet(Dataset):
    def __init__(self, imgs, segs, split='train'):

        self.root_segs = root_segs
        self.root_imgs = root_imgs
        self.dl = pascalVOCContextLoader(root_imgs, root_segs, split=split)

        self.reshaper = iaa.Noop()
        self.augmentations = iaa.Noop()

        self.augmentations = iaa.Sequential([
            iaa.Flipud(p=0.5),
            iaa.Fliplr(p=.5),
            iaa.Fliplr(p=.5),
            iaa.Rotate([360 / 4 * i for i in range(4)])
        ])

        self.reshaper = iaa.size.Resize(512)

        self.normalization = iaa.Sequential([
            rescale_augmenter,
            Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

        self.or_cntr_path = os.path.join(os.path.split(self.root_segs)[0], 'orientated_contours')
        
        self.prepare_all()

    def prepare_all(self):
        if (not os.path.exists(self.or_cntr_path)):
            os.makedirs(self.or_cntr_path)
            print('preparing orientation maps to {}'.format(self.or_cntr_path))

            dl = pascalVOCContextLoader(self.root_imgs, self.root_segs)
            for s in dl.splits:
                dl.split = s
                for ii in tqdm(range(len(dl))):
                    s = dl[ii]
                    or_cntr = interpolate_to_polygon(s['labels']).astype(np.uint8)
                    or_cntr = sparse.csr_matrix(or_cntr)
                    sparse.save_npz(pjoin(self.or_cntr_path, s['fname'] + '.npz'), or_cntr)

    def __len__(self):
        return len(self.dl)

    def __getitem__(self, idx):
        sample = self.dl[idx]

        or_cntr = sparse.load_npz(pjoin(self.or_cntr_path, sample['base_name'] + '.npz'))
        sample['or_cntr'] = or_cntr.toarray()

        aug = iaa.Sequential([self.reshaper, self.augmentations, self.normalization])
        aug_det = aug.to_deterministic()

        sample['or_cntr'] = ia.SegmentationMapsOnImage(sample['or_cntr'], shape=sample['or_cntr'].shape)
        sample['image'] = aug_det(image=sample['image'])
        sample['or_cntr'] = aug_det(segmentation_maps=sample['or_cntr']).get_arr()[..., None]
        sample['cntr'] = sample['or_cntr'].astype(bool)

        return sample

    @staticmethod
    def collate_fn(data):

        to_collate = ['image', 'or_cntr', 'cntr']

        out = dict()
        for k in data[0].keys():
            if (k in to_collate):
                out[k] = torch.stack([
                    torch.from_numpy(np.rollaxis(data[i][k], -1)).float()
                    for i in range(len(data))
                ])
            else:
                out[k] = [data[i][k] for i in range(len(data))]

        return out
        
        
def interpolate_to_polygon(arr, n_pts=10000, n_bins=8):
    # arr is an integer array
    contours = np.zeros(arr.shape)
    for c in np.unique(arr):
        arr_ = arr == c
        # label = measure.label(arr_)
        # for l in np.unique(label):
        #     if (l > 0):
        contours_, _ = cv2.findContours(
            (arr_).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours_ = np.squeeze(contours_[np.argmax([len(c)
        #                                             for c in contours_])])

        for cntr in contours_:
            if (cntr.shape[0] > 3):
                pts_contour = cntr.squeeze()
                # switch to x,y reference
                x = pts_contour[:, 0]
                y = arr.shape[0] - pts_contour[:, 1]
                # y = pts_contour[:, 1]
                bins = bin_contour(x, y, n_bins=n_bins, n_pts=n_pts)
                i, j = arr.shape[0] - y, x
                i = np.clip(i, 0, arr.shape[0] - 1)
                j = np.clip(j, 0, arr.shape[1] - 1)
                contours[i, j] = bins + 1

    # remove edges at borders
    contours[0, :] = 0
    contours[-1, :] = 0
    contours[:, 0] = 0
    contours[:, -1] = 0

    return contours


def contours_to_pts(x, y, n_pts=100):
    pts = np.concatenate((x[..., None], y[..., None]), axis=1)
    distance = np.cumsum(np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    interpolator = interp1d(distance, pts, kind='linear', axis=0)
    alpha = np.linspace(0, 1, n_pts)

    interp_pts = interpolator(alpha)
    interp_pts = np.concatenate((interp_pts, interp_pts[0, :][None, ...]))

    return interp_pts[:, 0], interp_pts[:, 1]


def segments_to_angles(x, y):
    pts = np.concatenate((x[..., None], y[..., None]), axis=1)
    dx = pts[:-1, 0] - pts[1:, 0]
    dy = pts[:-1, 1] - pts[1:, 1]
    tan = dy / (dx + 1e-8)
    angles = np.arctan(tan)
    # angles = np.unwrap(angles, np.pi / 2)
    # angles[angles < 0] = angles[angles < 0] + np.pi / 2
    # angles = np.arctan2(dy, dx)

    return angles


def bin_contour(x, y, n_bins=8, n_pts=10000):
    pts = np.concatenate((x[..., None], y[..., None]), axis=1)
    x_interp, y_interp = contours_to_pts(x, y, n_pts=n_pts)

    # calculate mid-points of each segment
    pts_interp = np.concatenate((x_interp[..., None], y_interp[..., None]),
                                axis=1)
    vec = pts_interp[1:, :] - pts_interp[:-1, :]
    M = pts_interp[:-1, :] + vec

    seg_idx, _ = pairwise_distances_argmin_min(pts, M)

    angles = segments_to_angles(x_interp, y_interp)
    inds = bin_angles(angles, n_bins=8)

    bins = inds[seg_idx]
    return bins


def bin_angles(angles, n_bins=8):

    # shift to [0, pi]
    angles += np.pi / 2
    angles -= np.pi / n_bins / 2

    bins = np.linspace(0, np.pi - np.pi / n_bins, n_bins)

    inds = np.digitize(angles, bins)
    inds[inds == n_bins] = 0
    return inds


void_fun = lambda x, random_state, parents, hooks: x


def rescale_images(images, random_state, parents, hooks):

    result = []
    for image in images:
        image_aug = np.copy(image)
        if (image.dtype == np.uint8):
            image_aug = image_aug / 255
        result.append(image_aug)
    return result


rescale_augmenter = iaa.Lambda(func_images=rescale_images,
                               func_segmentation_maps=void_fun,
                               func_heatmaps=void_fun,
                               func_keypoints=void_fun)


class Normalize(Augmenter):
    def __init__(self, mean, std, name=None, random_state=None):
        super(Normalize, self).__init__(name=name, random_state=random_state)
        self.mean = mean
        self.std = std
        self.n_chans = len(self.mean)

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        for i in range(nb_images):
            if (images[i].dtype == np.uint8):
                images[i] = images[i] / 255
            images[i] = [(images[i][..., c] - self.mean[c]) / self.std[c]
                         for c in range(self.n_chans)]

            images[i] = np.moveaxis(np.array(images[i]), 0, -1)
            images[i] = images[i].astype(float)
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def get_parameters(self):
        return [self.mean, self.std]

class pascalVOCContextLoader:
    """Data loader for the Pascal VOC semantic segmentation dataset.

    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.

    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.

    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """
    def __init__(self, root_imgs, root_segs, split='train'):
        self.root_imgs = root_imgs
        self.root_segs = root_segs

        self.splits = ['train', 'val', 'test']
        self.split = split

        self.all_base_names_ctxt = [
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(pjoin(self.root_segs, '*.mat'))
        ]

        # read pascal train and validation sets
        with open(pjoin(root_imgs, 'ImageSets', 'Main', 'train.txt')) as f:
            self.pascal_train = f.readlines()
        self.pascal_train = [x.strip() for x in self.pascal_train]
        with open(pjoin(root_imgs, 'ImageSets', 'Main', 'val.txt')) as f:
            self.pascal_val = f.readlines()
        self.pascal_val = [x.strip() for x in self.pascal_val]

        self.base_names = dict()
        self.base_names['train'] = [
            f for f in self.all_base_names_ctxt if f in self.pascal_train
        ]
        self.base_names['valtest'] = [
            f for f in self.all_base_names_ctxt if f in self.pascal_val
        ]

        self.base_names['val'] = self.base_names[
            'valtest'][:len(self.base_names['valtest']) // 2]
        self.base_names['test'] = self.base_names['valtest'][
            len(self.base_names['valtest']) // 2:]

    def __len__(self):
        return len(self.base_names[self.split])

    def __getitem__(self, index):
        base_name = self.base_names[self.split][index]
        im_path = pjoin(self.root_imgs, 'JPEGImages', base_name + '.jpg')
        lbl_path = pjoin(self.root_segs, base_name + '.mat')

        im = imread(im_path)
        data = io.loadmat(lbl_path)
        lbl = data['LabelMap']

        return {'image': im, 'labels': lbl, 'base_name': base_name}