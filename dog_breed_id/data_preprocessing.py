# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_data_preprocessing.ipynb.

# %% auto 0
__all__ = ['annotated_image', 'get_breedname', 'resize_bboxes', 'save_resized', 'get_format_from', 'get_annotations_path_from',
           'get_resized_bboxes', 'plot_random_images', 'get_cat_id_mappings', 'get_image_id_mappings', 'get_cats_json',
           'get_images_json', 'bbox_to_coco', 'coco_to_bbox', 'get_annotations_json', 'to_coco']

# %% ../nbs/02_data_preprocessing.ipynb 6
import cv2
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
from miniai.datasets import show_images

# %% ../nbs/02_data_preprocessing.ipynb 14
def annotated_image(image, boxes):
    """
    image - image to be annotated
    boxes - list of bounding boxes to be annotated
    """
    imann = image.copy()
    for box in boxes:
        cv2.rectangle(imann, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 255, 0), 2)
    return imann

# %% ../nbs/02_data_preprocessing.ipynb 18
def get_breedname(path, normalize=False):
    """
    impath - path to image or annotation
    normalize - if set to true, lowercases the name
    returns the name of the dog breed
    """
    import os
    parts = path.split(os.sep)
    breed = parts[-2].split('-')[-1]
    return breed if not normalize else breed.lower()

# %% ../nbs/02_data_preprocessing.ipynb 25
def resize_bboxes(boxes, src_size, dst_size):
    """
    boxes - Bounding boxes in src image
    src_size - size of the src image
    dst_size - size of the dst image
    """
    boxes = np.array(boxes).astype(np.float64)
    fx = dst_size[1] / float(src_size[1])
    fy = dst_size[0] / float(src_size[0])
    boxes[:, [0, 2]] *= fx
    boxes[:, [1, 3]] *= fy
    return boxes

# %% ../nbs/02_data_preprocessing.ipynb 28
def save_resized(impath, dsize=(256, 256)):
    """
    impath - parent path to images
    dsize - destination image size
    Saves images under the parent folder of impaths under new folder `resized`
    """
    import glob
    import os
    images = glob.glob(f'{impath}/**/*.jpg') + glob.glob(f'{impath}/**/*.jpeg')
    dstdir = (Path(impath).parent/'resized')
    dstdir.mkdir(parents=True, exist_ok=True)    
    for image in images:
        uniquename = os.sep.join(image.split(os.sep)[-2:])
        img = cv2.imread(image)
        imgr = cv2.resize(img, dsize)
        dst = f'{str(dstdir)}/{uniquename}'
        dst_image_parent = Path(dst).parent
        dst_image_parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(dst, imgr)

# %% ../nbs/02_data_preprocessing.ipynb 31
def get_format_from(impath):
    """Returns one of 'stanford' or 'tsinghua'"""
    return 'stanford' if 'stanford' in str(impath) else 'tsinghua'

def get_annotations_path_from(impath):
    impath = Path(impath)
    format = get_format_from(impath)
    annotations_folder = 'low-annotations' if format == 'tsinghua' else 'Annotation'
    annotations_path = impath.parent.parent.parent/annotations_folder
    filename = f'{impath.parent.stem}/{impath.stem}'
    extension = str(impath).split('.')[-1]
    annotation = annotations_path/filename
    if format == 'tsinghua':
        annotation = Path(str(annotation) + f'.{extension}.xml')
    return annotation

def get_resized_bboxes(impath, dsize=(256, 256)):
    """
    Gets the bounding boxes of the image given a specified destination resize which
    assumes the image has been resized and is not the original size as provided
    in the annotations file.
    impath - path to the image on disk
    annotations_path - path to corresponding annotations folder
    dsize - Size of the resized image
    format - one of 'stanford' or 'tsinghua'
    """
    format = get_format_from(impath)
    assert format in ['stanford', 'tsinghua'], 'Format neither "stanford" nor "tsinghua"'
    import os
    import xml.etree.ElementTree as ET
    parts = impath.split(os.sep)
    imname = Path(os.sep.join(parts[-2:]))
    uniquename = imname.parent/imname.stem
    #annotation_file = f'{annotations_path}/{uniquename}'
    annotation_file = get_annotations_path_from(impath)
    root = ET.parse(annotation_file)
    boxes = []
    bndbox_index = 4
    src_w = float(root.find('size')[0].text)
    src_h = float(root.find('size')[1].text)
    bndbox = 'bndbox' if format=='stanford' else 'bodybndbox'
    for obj in root.iter('object'):
        boxelem = obj.find(bndbox)
        left, top, right, bottom = boxelem[0].text, boxelem[1].text, boxelem[2].text, boxelem[3].text
        left, top, right, bottom = map(int, [left, top, right, bottom])
        boxes.append([left, top, right, bottom])
    return resize_bboxes(boxes, (src_h, src_w), dsize)

# %% ../nbs/02_data_preprocessing.ipynb 34
def plot_random_images(datadir, n=16):
    """
    datadir - Parent folder where both stanford and tsinghua datasets are
    """
    import glob
    files = glob.glob(f'{datadir}/**/resized/**/*.jpeg') + glob.glob(f'{datadir}/**/resized/**/*.jpg')
    files = np.random.choice(files, size=(n,), replace=False)
    bboxes = []
    format = 'stanford'
    annotations = f'{datadir}/stanford_dogs/Annotation'
    for filepath in files:
        if 'tsinghua' in filepath:
            format = 'tsinghua'
            annotations = f'{datadir}/tsinghua_dogs/Low-Annotations'
        else:
            format = 'stanford'
            annotations = f'{datadir}/stanford_dogs/Annotation'
        boxes = get_resized_bboxes(filepath)
        bboxes.append(boxes)
    images = [np.array(Image.open(filepath)) for filepath in files]
    titles = [get_breedname(filepath, normalize=True) for filepath in files]
    annotated_images = [annotated_image(im, bbox) for (im, bbox) in zip(images, bboxes)]
    show_images(annotated_images, titles=titles)

# %% ../nbs/02_data_preprocessing.ipynb 36
def get_cat_id_mappings(paths):
    """
    paths - List of folder paths with images
    In this case it will be Stanford and Tsinghua paths
    Returns a mapping of each breed to a numerical Id
    
    """
    id2cats = {}
    cats2ids = {}
    import glob
    files = []
    for path in paths:
        files.extend(glob.glob(f'{path}/**/resized/**/*.jpg') + glob.glob(f'{path}/**/resized/**/*.jpeg'))
    count = 0
    for file in files:
        breed = get_breedname(file, normalize=True)
        if breed not in cats2ids:
            cats2ids[breed] = count
            count += 1
    ids2cats = {i:c for (c, i) in cats2ids.items()}
    return ids2cats, cats2ids

# %% ../nbs/02_data_preprocessing.ipynb 38
def get_image_id_mappings(paths):
    """Gets back and forth mappings between the absolute image paths and ids
    paths - list of data paths to process.
    """
    import glob
    ims2ids = {}
    ids2ims = {}
    files = []
    for path in paths:
        files.extend(glob.glob(f'{path}/**/resized/**/*.jpg') + glob.glob(f'{path}/**/resized/**/*.jpeg'))
    count = 0
    for file in files:
        if file not in ims2ids:
            # Ensure annotations exist
            if Path(get_annotations_path_from(file)).exists():
                ims2ids[file] = count
                count += 1
    ids2ims = {i:f for f,i in ims2ids.items()}
    return ids2ims, ims2ids

# %% ../nbs/02_data_preprocessing.ipynb 42
import json
from pycocotools import coco

def get_cats_json(cats2ids):
    cats = []
    for cat, id in cats2ids.items():
        cats.append(
            {
                'id': id,
                'name': cat
            }
        )
    return cats

def get_images_json(ims2ids, dsize=(256, 256)):
    images = []
    for image, id in ims2ids.items():
        if dsize is None:
            h, w = cv2.imread(image).shape[:2]
            dsize = (h, w)
        images.append(
            {
                'file_name': image,
                'height': dsize[0],
                'width': dsize[1],
                'id': id
            }
        )
    return images

def bbox_to_coco(bbox):
    return np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]) # left, top, w, h from left, top, right, bottom

def coco_to_bbox(bbox):
    np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) # left, top, right, bottom from left, top, w, h

def get_annotations_json(id, image_id, cat_id, bbox):
    return {
        'id': id,
        'image_id': image_id,
        'category_id': cat_id,
        'bbox': bbox_to_coco(bbox).tolist()
    }

def to_coco(dataroot, ims2ids, cats2ids):
    """Rewrites the datasets (stanford and tsinghua) located at data root into format expected by COCO. 
    dataroot - Root of the data folder containing both Stanford and Tsinghua datasets. **It is assumed 
    that all files have already been resized and there are resized folders for both stanford and tsinghua.
    ims2ids - Unique mapping of each file to a numeric id
    cats2ids - Unique mapping of each breed (normalized to lower case) to a numeric id
    that the resized data is already present. """
    categories = get_cats_json(cats2ids)
    images = get_images_json(ims2ids)
    ann_count = 0
    annotations = []
    for im, id in ims2ids.items():
        annotations_path = get_annotations_path_from(im)
        category = get_breedname(im, normalize=True)
        cat_id = cats2ids[category]
        image_id = id
        bboxes = get_resized_bboxes(im)
        for box in bboxes:
            annotation = get_annotations_json(ann_count, image_id, cat_id, box)        
            ann_count += 1
    return {
        'categories': categories,
        'images': images,
        'annotations': annotations
    }
