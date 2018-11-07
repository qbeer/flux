"""
Parsing code for the 
"""
import pickle, os
import numpy as np
from tqdm import tqdm
from typing import List
import xml.dom.minidom as minidom
import scipy
import tensorflow as tf

from flux.util.logging import log_message, log_warning
from flux.backend.data import maybe_download_and_store_tar, register_to_datastore, retrieve_subkeys
from flux.backend.globals import DATA_STORE
from flux.processing.vision.util import load_image, encode_jpeg, encode_png

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def image_path_from_index(index, src, end=".jpg"):
        img_path = os.path.join(src, index+end)
        return img_path

class PascalVOC:
    """Class for the Pascal VOC 2012 dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """

    def __init__(self, force_rebuild: bool=False, nohashcheck: bool=True) -> None:
        tar_file = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
        self._classes = ('__background__', # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes = 21
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
        self.max_num_obj = 50
        self.voc_root_key = "pascal/voc/2012"
        self.file_structure = os.path.join("VOCtrainval_11-May-2", "VOCdevkit", "VOC2012")
        
        work_file_path = os.path.join(DATA_STORE.working_directory, self.file_structure)
        _annotation_path = os.path.join(work_file_path, "Annotations/")
        _problems = os.path.join(work_file_path, "ImageSets/")
        _images = os.path.join(work_file_path, "JPEGImages/")
        _segmentation_class = os.path.join(work_file_path, "SegmentationClass/")
        _segmentation_object = os.path.join(work_file_path, "SegmentationObject/")
        self.annotation_key = os.path.join(self.voc_root_key, "annotations")
        self.images_key = os.path.join(self.voc_root_key, "images")
        self.segmentation_key = os.path.join(self.voc_root_key, "segmentation", "class")
        self.segmentation_obj_key = os.path.join(self.voc_root_key, "segmentation", "obj")
        self.problems_key = os.path.join(self.voc_root_key, "ImageSets")
        if force_rebuild:
            log_message("Copying data to destination folder in flux")
            self.raw_data_key = maybe_download_and_store_tar(url=tar_file, root_key='pascal/voc/2012', use_subkeys=False)
            self.images_key = DATA_STORE.add_folder(self.images_key, _images)
            self.segmentation_key = DATA_STORE.add_folder(self.segmentation_key, _segmentation_class)
            self.segmentation_obj_key = DATA_STORE.add_folder(self.segmentation_obj_key, _segmentation_object)
            self.annotation_key = DATA_STORE.add_folder(self.annotation_key, _annotation_path)
            self.problems_key = DATA_STORE.add_folder(self.problems_key, _problems)
        problems_path = os.path.join(DATA_STORE.root_filepath, self.voc_root_key, "VOCdevkit", "VOC2012", "ImageSets")

        self.problems = [name for name in os.listdir(problems_path)]
        self.image_path = DATA_STORE[self.images_key]
        self.annotation_path = DATA_STORE[self.annotation_key]
        self.seg_class_path = DATA_STORE[self.segmentation_key]
        self.seg_obj_path = DATA_STORE[self.segmentation_obj_key]

    def _load_pascal_annotation(self, index):
            """
            Load image and bounding boxes info from XML file in the PASCAL VOC
            format.
            src: https://github.com/rbgirshick/fast-rcnn/blob/master/lib/datasets/pascal_voc.py
            """
            filename = os.path.join(self.annotation_path, index+".xml")
            def get_data_from_tag(node, tag):
                return node.getElementsByTagName(tag)[0].childNodes[0].data

            with open(filename) as f:
                data = minidom.parseString(f.read())

            objs = data.getElementsByTagName('object')
            num_objs = len(objs)

            boxes = np.zeros((self.max_num_obj, 4), dtype=np.int64)
            gt_classes = np.zeros((self.max_num_obj), dtype=np.int64)
            overlaps = np.zeros((self.max_num_obj, self.num_classes), dtype=np.float32)

            # Load object bounding boxes into a data frame.
            for ix, obj in enumerate(objs):
                # Make pixel indexes 0-based
                x1 = float(get_data_from_tag(obj, 'xmin')) - 1
                y1 = float(get_data_from_tag(obj, 'ymin')) - 1
                x2 = float(get_data_from_tag(obj, 'xmax')) - 1
                y2 = float(get_data_from_tag(obj, 'ymax')) - 1
                cls = self._class_to_ind[
                        str(get_data_from_tag(obj, "name")).lower().strip()]
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
            overlaps = scipy.sparse.csr_matrix(overlaps)
            overlaps = overlaps.toarray().flatten()
            return {'boxes' : _int64_features(boxes.flatten()),
                    'num_objs': _int64_features([num_objs]),
                    'gt_classes': _int64_features(gt_classes.flatten()),
                    'gt_overlaps' : _float_features(overlaps),
                    'flipped' : _int64_features([int(False)])}

class PascalVOC_Segmentation(PascalVOC):

    def __init__(self, force_rebuild=False, num_parallel_reads=1):
        super(PascalVOC_Segmentation, self).__init__(force_rebuild=force_rebuild)
        self.problem_name = "Segmentation"
        self.problems = [problems for problems in self.problems if self.problem_name in problems]

        self.train_tf_key = os.path.join(self.voc_root_key, self.problem_name.lower(), "tfrecord", "train")
        self.val_tf_key = os.path.join(self.voc_root_key, self.problem_name.lower(), "tfrecord", "val")

        if force_rebuild:
            self.num_train_examples = self._build_dataset(dataset="train")
            self.num_val_examples = self._build_dataset(dataset="val")
            self.train_path = DATA_STORE[self.train_tf_key]
            self.val_path = DATA_STORE[self.val_tf_key]
        else:
            self.train_path = DATA_STORE[self.train_tf_key]
            self.val_path = DATA_STORE[self.val_tf_key]
            self.num_train_examples = sum(1 for _ in tf.python_io.tf_record_iterator(self.train_path))
            self.num_val_examples = sum(1 for _ in tf.python_io.tf_record_iterator(self.val_path))
        # Setup some default options for the dataset
        self._val_db = None
        self._train_db = None
        self._test_db = None
        self.num_parallel_reads = num_parallel_reads

    @property
    def train_db(self,):
        if self._train_db is None:
            self._train_db = tf.data.TFRecordDataset(self.train_path).map(self._map_fn, num_parallel_calls=self.num_parallel_reads)
        return self._train_db  

    @property
    def val_db(self,):
        if self._val_db is None:
            self._val_db = tf.data.TFRecordDataset(self.val_path).map(self._map_fn, num_parallel_calls=self.num_parallel_reads)
        return self._val_db  

    def _map_fn(self, serialized_example):

        feature_dict = {
            'boxes': tf.FixedLenFeature([self.max_num_obj*4], tf.int64),
            'gt_classes': tf.FixedLenFeature([self.max_num_obj], tf.int64),
            'gt_overlaps': tf.FixedLenFeature([self.max_num_obj*self.num_classes], tf.float32),
            'flipped': tf.FixedLenFeature([1], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'seg_class': tf.FixedLenFeature([], tf.string),
            'seg_obj': tf.FixedLenFeature([], tf.string),
            'num_objs': tf.FixedLenFeature([1], tf.int64)
        } 

        features = tf.parse_single_example(
            serialized_example,
            features=feature_dict)

        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.cast(image, tf.float32) / 255.0

        seg_class = tf.image.decode_png(features['seg_class'], channels=3)
        seg_class = tf.cast(seg_class, tf.float32) / 255.0

        seg_obj = tf.image.decode_png(features['seg_obj'], channels=3)
        seg_obj = tf.cast(seg_obj, tf.float32) / 255.0

        boxes = tf.reshape(features['boxes'], (self.max_num_obj, 4))
        gt_classes = tf.reshape(features['gt_classes'], (self.max_num_obj,))
        gt_overlaps = tf.reshape(features['gt_overlaps'], (self.max_num_obj, self.num_classes))
        flipped = features["flipped"]
        num_objs = features["num_objs"]
        return (image,
                seg_class,
                seg_obj,
                boxes,
                gt_classes,
                gt_overlaps,
                flipped,
                num_objs)

    def _build_dataset(self, dataset):
        _problem_key = self.problems[0]
        if len(_problem_key) < 1:
            log_warning("Problem key doesn't exist for {}.  ".format(dataset) + str(_problem_key))
            raise EnvironmentError()
        problem_path = os.path.join(DATA_STORE[self.problems_key], _problem_key, dataset+".txt")
        
        tf_record_key = self.train_tf_key if dataset == "train" else self.val_tf_key       
        log_message("Retrieving the index from " + problem_path)
        assert(os.path.exists(problem_path))
        with open(problem_path, 'r') as f:
            images_index = [x.strip() for x in f.readlines()]
        tf_record_writer = tf.python_io.TFRecordWriter(DATA_STORE.create_key(tf_record_key, 'data.tfrecords', force=True))

        errors = 0
        log_message("Building {} dataset...".format(dataset))
        total_num_examples = 0
        for idx, index in tqdm(enumerate(images_index)):
            img_path = image_path_from_index(index, self.image_path, '.jpg')
            feature_dict = self._load_pascal_annotation(index)

            image = load_image(img_path)
            image = encode_jpeg(image)
            if image is None:
                errors += 1
                log_warning('Error loading image: {}. {} Errors so far.'.format(img_path, errors))
                continue

            seg_cls_path = image_path_from_index(index, self.seg_class_path, '.png')
            seg_class = load_image(seg_cls_path)
            seg_class = encode_png(seg_class)
            seg_obj_path = image_path_from_index(index, self.seg_obj_path, '.png')
            seg_obj = load_image(seg_obj_path)
            seg_obj = encode_png(seg_obj)

            if seg_class is None:
                errors += 1
                log_warning('Error loading image: {}. {} Errors so far.'.format(seg_cls_path, errors))
                continue
            if seg_obj is None:
                errors += 1
                log_warning('Error loading image: {}. {} Errors so far.'.format(seg_obj_path, errors))
                continue
            feature_dict["image"] =  _bytes_feature(tf.compat.as_bytes(image))
            feature_dict["seg_class"] =  _bytes_feature(tf.compat.as_bytes(seg_class))
            feature_dict["seg_obj"] =  _bytes_feature(tf.compat.as_bytes(seg_obj))

            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            tf_record_writer.write(example.SerializeToString())
            total_num_examples += 1
        tf_record_writer.close()
        DATA_STORE.update_hash(tf_record_key)
        return total_num_examples