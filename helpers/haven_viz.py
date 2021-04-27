import os
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries

def get_image(img, denorm=None, size=None, points=None, radius=10,
              mask=None, heatmap=None):
    return save_image(None, img, denorm, size, points, radius,
               mask, heatmap, return_image=True)

def save_image(fname, img, denorm=None, size=None, points=None, radius=10,
               mask=None, heatmap=None, makedirs=True, return_image=False, nrow=8):
    """Save an image into a file.
    Parameters
    ----------
    fname : str
        Name of the file
    img : [type]
        Image data. #TODO We asume it is.....?????? in [0, 1]? Numpy? PIL? RGB?
    makedirs : bool, optional
        If enabled creates the folder for saving the file, by default True
    """
    if not isinstance(img, torch.Tensor):
        if img.min() >= 0 and img.max() > 1:
            img = img / 255.
        img = torch.as_tensor(img)
    if img.ndim == 4:
        img = torchvision.utils.make_grid(img, nrow=nrow)
    if denorm:
        img = denormalize(img, mode=denorm)
    if points is not None:
        if isinstance(img, np.ndarray):
            img = torch.FloatTensor(img)
        img = img.squeeze()
        if img.ndim == 2:
            img = img[None].repeat(3,1,1)
        y_list, x_list = np.where(points.squeeze())
        c_list = []
        for y, x in zip(y_list, x_list):
            c_list += [points.squeeze()[y, x]]
        img = points_on_image(y_list, x_list, img, 
                 radius=radius, c_list=c_list)

    if mask is not None:
        img = mask_on_image(mask, img)

    if img.dtype == 'uint8':
        img = Image.fromarray(img)
    else:
        arr = f2l(t2n(img)).squeeze()
        # print(arr.shape)
        if size is not None:  
            arr = Image.fromarray(arr)
            arr = arr.resize(size)
            arr = np.array(arr)

        img = Image.fromarray(np.uint8(arr * 255))
    if return_image:
        return img

    if fname is not None:
        dirname = os.path.dirname(fname)
        if makedirs and dirname != '':
            os.makedirs(dirname, exist_ok=True)
        img.save(fname)

def mask_on_image(image, mask, add_bbox=False, return_pil=False):
    """[summary]
    
    Parameters
    ----------
    image : [type]
        [description]
    mask : [type]
        [description]
    add_bbox : bool, optional
        [description], by default True
    
    Returns
    -------
    [type]
        [description]
    """
    image = image_as_uint8(image)
    mask = np.array(mask).squeeze()
    obj_ids = np.unique(mask)

    # polygons = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1][0]
    red = np.zeros(image.shape, dtype='uint8')
    red[:,:,2] = 255
    alpha = 0.5
    result = image.copy()
    for o in obj_ids:
        if o == 0:
            continue
        ind = mask==o
        result[ind] = result[ind] * alpha + red[ind] * (1-alpha)
        pos = np.where(ind)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        if add_bbox:
            result = cv2.rectangle(result, (xmin, ymin), 
                                                    (xmax, ymax), 
                                                    color=(0,255,0), 
                                                    thickness=2)
    result = mark_boundaries(result, mask) 

    if return_pil:
        return Image.fromarray(result)

    return result

def text_on_image(text, image):
    """Adds test on the image
    
    Parameters
    ----------
    text : [type]
        [description]
    image : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,40)
    fontScale              = 0.8
    fontColor              = (1,1,1)
    #lineType               = 1 unused var
    # img_mask = skimage.transform.rescale(np.array(img_mask), 1.0)
    # img_np = skimage.transform.rescale(np.array(img_points), 1.0)
    img_np = cv2.putText(image, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness=2
        # lineType
        )
    return img_np

def points_on_image(y_list, x_list, image, radius=3, c_list=None):
    """[summary]
    
    Parameters
    ----------
    y_list : [type]
        [description]
    x_list : [type]
        [description]
    image : [type]
        [description]
    radius : int, optional
        [description], by default 3
    
    Returns
    -------
    [type]
        [description]
    """
    image_uint8 = image_as_uint8(image)

    H, W, _ = image_uint8.shape
    color_list = [(255, 0, 0) , (0, 255, 0) , (0, 0, 255) ]
    for i, (y, x) in enumerate(zip(y_list, x_list)):
        if y < 1:
            x, y = int(x*W), int(y*H) 
        else:
            x, y = int(x), int(y) 
            
        # Blue color in BGR 
        if c_list is not None:
            color = color_list[c_list[i]] 
        else:
            color = color_list[0] 
        
        # Line thickness of 2 px 
        thickness = 2
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        image_uint8 = cv2.circle(image_uint8, (x,y), 2, color, thickness) 

        start_point = (x-radius*2, y-radius*2) 
        end_point = (x+radius*2, y+radius*2) 
        thickness = 2
        color = (255, 0, 0)
        
        #image_uint8 = cv2.rectangle(image_uint8, start_point, end_point, color, thickness) 

    return image_uint8 / 255.

def image_as_uint8(img):
    """Returns a uint8 version of the image
    
    Parameters
    ----------
    img : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    image = f2l(np.array(img).squeeze())
    
    if image.dtype != 'uint8':
        image_uint8 = (image*255).astype("uint8").copy()
    else:
        image_uint8 = image 

    return image_uint8


def denormalize(img, mode=0):  # TODO: Remove the default value or set to a valid number, complete documentation
    """Denormalize an image.
    Parameters
    ----------
    img : [type]
        Input image to denormalize
    mode : int or str, optional
        Predefined denormalizations, by default 0
        If 1 or 'rgb'... 
        If 2 or 'brg'...,
        If 3 or 'basic'...
        Else do nothing
    Returns
    -------
    [type]
        Denormalized image
    """
    # _img = t2n(img)
    # _img = _img.copy()
    image = t2n(img).copy().astype("float")

    if mode in [1, "rgb"]:
        mu = np.array([0.485, 0.456, 0.406])
        var = np.array([0.229, 0.224, 0.225])
        image = _denorm(image, mu, var)

    elif mode in [2, "bgr"]:
        mu = np.array([102.9801, 115.9465, 122.7717])
        var = np.array([1, 1, 1])
        image = _denorm(image, mu, var, bgr2rgb=True).clip(0, 255).round()

    elif mode in [3, "basic"]:
        mu = np.array([0.5, 0.5, 0.5])
        var = np.array([0.5, 0.5, 0.5])
        image = _denorm(image, mu, var)

    # TODO: Add a case for 0 or None and else raise an error exception.

    return image

def _denorm(image, mu, var, bgr2rgb=False):
    """Denormalize an image.
    Parameters
    ----------
    image : [type]
        Image to denormalize
    mu : [type]
        Mean used to normalize the image
    var : [type]
        Variance used to normalize the image
    bgr2rgb : bool, optional
        Whether to also convert from bgr 2 rgb, by default False
    Returns
    -------
    [type]
        Denormalized image
    """
    if image.ndim == 3:
        result = image * var[:, None, None] + mu[:, None, None]  # TODO: Is it variance or std?
        if bgr2rgb:
            result = result[::-1]
    else:
        result = image * var[None, :, None, None] + mu[None, :, None, None]
        if bgr2rgb:
            result = result[:, ::-1]
    return result

def t2n(x):
    """Pytorch tensor to Numpy array.
    Parameters
    ----------
    x : Pytorch tensor
        A Pytorch tensor to transform
    Returns
    -------
    Numpy array
        x transformed to numpy array
    """
    try:
        x = x.detach().cpu().numpy()
    except:
        x = x

    return x


def l2f(X):
    """Move the channels from the last dimension to the first dimension.
    Parameters
    ----------
    X : Numpy array
        Tensor with the channel dimension at the last dimension
    Returns
    -------
    Numpy array
        X transformed with the channel dimension at the first dimension
    """
    if X.ndim == 3 and (X.shape[0] == 3 or X.shape[0] == 1):
        return X
    if X.ndim == 4 and (X.shape[1] == 3 or X.shape[1] == 1):
        return X

    if X.ndim == 4 and (X.shape[1] < X.shape[3]):
        return X

    # Move the channel dimension from the last position to the first one
    if X.ndim == 3:
        return np.transpose(X, (2, 0, 1))
    if X.ndim == 4:
        return np.transpose(X, (0, 3, 1, 2))

    return X

def f2l(X):
    """Move the channels from the first dimension to the last dimension.
`   Parameters
    ----------
    X : Numpy array
        Tensor with the channel dimension at the first dimension
    Returns
    -------
    Numpy array
        X transformed with the channel dimension at the last dimension
    """
    if X.ndim == 3 and (X.shape[2] == 3 or X.shape[2] == 1):
        return X
    if X.ndim == 4 and (X.shape[3] == 3 or X.shape[3] == 1):
        return X

    # Move the channel dimension from the first position to the last one
    if X.ndim == 3:
        return np.transpose(X, (1, 2, 0))
    if X.ndim == 4:
        return np.transpose(X, (0, 2, 3, 1))

    return X

def gray2cmap(gray, cmap="jet", thresh=0):
    """gets a heatmap for a given gray image. Can be used to visualize probabilities.
    
    Parameters
    ----------
    gray : [type]
        [description]
    cmap : str, optional
        [description], by default "jet"
    thresh : int, optional
        [description], by default 0
    
    Returns
    -------
    [type]
        [description]
    """
    # Gray has values between 0 and 255 or 0 and 1
    gray = t2n(gray)
    gray = gray / max(1, gray.max())
    gray = np.maximum(gray - thresh, 0)
    gray = gray / max(1, gray.max())
    gray = gray * 255

    gray = gray.astype(int)
    #print(gray)

    from matplotlib.cm import get_cmap
    cmap = get_cmap(cmap)

    output = np.zeros(gray.shape + (3, ), dtype=np.float64)

    for c in np.unique(gray):
        output[(gray == c).nonzero()] = cmap(c)[:3]

    return l2f(output)
