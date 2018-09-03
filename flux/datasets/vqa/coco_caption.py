"""
Wrapper for MSCOCO Dataset

"""

from flux.backend.data import maybe_download_and_store_zip, write_csv_file
from flux.backend.globals import DATA_STORE
import os
import json
import numpy as np
from typing import Dict, List
from PIL import Image

class CocoCaption:
    """
    Write the version and year of this data: TODO (Karen)
    """
    URLS = dict(annotations=
        dict(Train = "http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip", \
             Validate="http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip"),
    questions = dict(
        Train ="http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip", \
        Validate = "http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip"),
    images =  dict(
        Train ="http://images.cocodataset.org/zips/train2014.zip",
        Validate ="http://images.cocodataset.org/zips/val2014.zip"
    ))



    def __init__(self, validation_only:bool):

        # Training Data
        self.questions_train:List = None
        self.images_train:List = None
        self.annotations_train:List = None

        self._index:np.ndarray = None
        self._question_index:Dict = None
        self._answers_index:Dict = None
        self.questions_key:List = None
        self.is_training:bool = False

        self._annotations: List = None
        self._questions: List = None

        # Download Dataset.  
        self.annotations_val = maybe_download_and_store_zip(
                url=CocoCaption.URLS["annotations"]["Validate"], root_key='vqa/annotations-val')
        self.questions_val = maybe_download_and_store_zip(
                url=CocoCaption.URLS["questions"]["Validate"], root_key='vqa/questions-val')
        self.images_val = maybe_download_and_store_zip(
                url=CocoCaption.URLS["images"]["Validate"], root_key='vqa/images-val')

        if not validation_only:
            self.annotations_train = maybe_download_and_store_zip(
                    url=CocoCaption.URLS["annotations"]["Train"], root_key='vqa/annotations-train')
            self.questions_train = maybe_download_and_store_zip(
                    url=CocoCaption.URLS["questions"]["Train"], root_key='vqa/questions-train')  
            self.images_train = maybe_download_and_store_zip(
                    url=CocoCaption.URLS["images"]["Train"], root_key='vqa/images-train')  

    
    def load_questions(self, is_training:bool) -> List:
        # Read questions file, outputs a list of dictionaries
        if self._questions is not None:
            return self._questions
        questions_file = self.read_file_from_db(is_training, self.questions_train, self.questions_val)
        with open(questions_file) as j:
            question_dict = json.load(j)
        question_list = question_dict["questions"]

        # Outputs keys for each dictionary in question_list
        self.questions_key = list(question_dict.keys())
        self._questions = question_list
        return question_list

    def get_questions_key(self, is_training:bool) -> List:
        if self.questions_key is None:
            return self.load_questions(is_training)
        return self.questions_key

    def read_file_from_db(self, is_training:bool, train_key:List, val_key:List) -> str:
        if is_training:
            if train_key is None:
                raise Exception("Initialized with validation only.  Training data not downloaded.")
            return DATA_STORE.get_file(train_key[0])["fpath"]
        
        return DATA_STORE.get_file(val_key[0])["fpath"]

    def load_annotations(self, is_training:bool) -> List:
        # Read questions file, outputs a list of dictionaries
        if self._annotations is not None:
            return self._annotations
        annotation_file = self.read_file_from_db(is_training, self.annotations_train, self.annotations_val)
        with open(annotation_file) as j:
            ann_file = json.load(j)
        ann = ann_file["annotations"] # for our case, there are 443757 of them

        self._annotations = ann
        return ann

    def process_annotations(self, raw_annotation:List, only_yes:bool=True) -> Dict:
        # Create index on answers by question_id
        processed_annotations = {}
        for d in raw_annotation:
            q_id = d["question_id"]
            ans = [answer["answer_id"] for answer in d["answers"] if not only_yes or answer["answer_confidence"] == "yes"]
            processed_annotations[q_id] = ans
        return processed_annotations

    def build_index(self, is_training:bool, index_img_unique:bool=False) -> np.ndarray:
        filename = "train_im_qa_index.csv" if is_training else "val_im_qa_index.csv"
        csv_path = write_csv_file("vqa/im_qa_index", filename, "query for image, qa, and index")
        return self._bulid_index(is_training, csv_path, index_img_unique)

    def _bulid_index(self, is_training:bool, index_path:str, index_img_unique:bool=False) -> np.ndarray:
        # Build an index of [im_id, question_id, answer_id]
        if self._index is not None and is_training == self.is_training:
            return self._index
        if os.path.exists(index_path):
            self.is_training = is_training
            index_raw_data = self.read_csv(index_path)
            if index_img_unique:
                # TODO: unqiue image for all (q,a) pairs
                self.index = np.array([])
            else:
                self.index = index_raw_data
            return self.index
        else:
            self.is_training = is_training
            annotations = self.process_annotations(self.load_annotations(is_training))
            questions = self.load_questions(is_training)
            index = []
            for q in questions:
                q_id = q["question_id"]
                im_id = q["image_id"]
                for ans in annotations[q_id]:
                    index.append([im_id, q_id, ans])
            self.index = np.array(index, dtype=np.int64)
            self.write_csv(index_path, self.index)
            return self.index

    @property
    def question_index(self) -> Dict:
        # See _build_questions_index
        return self._build_questions_index()

    def _build_questions_index(self) -> Dict:
        # Create index between question_id: question
        if self._question_index is not None:
            return self._question_index
        questions = self.load_questions(self.is_training)
        self._question_index = {q["question_id"]:q["question"] for q in questions}
        return self._question_index

    def answer_index(self, only_yes:bool) -> Dict:
        return self._build_answers_index(only_yes)


    def _build_answers_index(self, only_yes:bool=True) -> Dict:
        if self._answers_index is not None:
            return self._answers_index
        self._answers_index = {}
        annotations = self.load_annotations(self.is_training)
        for a in annotations:
            for choice in a["answers"]:
                if not only_yes or choice["answer_confidence"] == "yes":
                    self._answers_index[(a["question_id"], choice["answer_id"])] = choice["answer"]
        return self._answers_index  

        
    def read_csv(self, csv_path: str) -> np.ndarray:
        return np.loadtxt(csv_path, dtype=np.int)
    
    def write_csv(self, csv_path: str, data: np.ndarray) -> None:
        assert len(data.shape) == 2
        assert data.shape[1] == 3

        np.savetxt(csv_path, data, fmt="%d")

    # def train_data_stream(self):
    #     if self.images_train is None:
    #         raise Exception("Download train_data_first by setting validation_only to False")
        
    #     if not self.is_training:
    #         raise Exception("Build index first with build_index(is_training=True)"
        
        
    #     return

    def val_data_stream(self):
        if self.images_val is None:
            raise Exception("Download val_data first by setting validation_only to True")
        
        for img_id, q_id, a_id in self.index:
            img_key = "vqa/images-val/val2014/COCO_val2014_%012d" % img_id
            img_file = DATA_STORE.get_file(img_key)
            img = self.image_from_file(img_file["fpath"])
            q = self.question_index[q_id]
            #TODO: Fix this hardcode
            a = self.answer_index(True)[(q_id, a_id)]
            yield (img, q, a)

    def image_from_file(self, fname:str) -> np.ndarray:
        img = Image.open(fname)
        return np.array(img)
