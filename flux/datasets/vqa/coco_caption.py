"""
Wrapper for MSCOCO Dataset

"""

from flux.backend.data import maybe_download_and_store_zip
from flux.backend.globals import DATA_STORE
import os
import json

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



    def __init__(self, validation_only, sample_size):

        # Metadata

        self.sample_size = sample_size
        
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

    def load_questions(self, is_training):
        questions_file = self.read_file_from_db(is_training, self.questions_train, self.questions_val)
        question_list = self.load_json(questions_file)["questions"]
        return question_list

    def read_file_from_db(self, is_training, train_key, val_key):
        if is_training:
            if train_key is None:
                raise Exception("Initialized with validation only.  Training data not downloaded.")
            return DATA_STORE[train_key]
        
        return DATA_STORE[val_key]

    def load_annotations(self, is_training):
        annotation_file = self.read_file_from_db(is_training, self.annotations_train, self.annotations_val)
        ann_file = self.load_json(annotation_file)
        ann = ann_file["annotations"] # for our case, there are 443757 of them
        return ann

    # def answers_from_image(self, ids):
    #     return [a["answers"]for a in self.annotations if a["image_id"] == ids]

    # def image_from_file(self, fname, prefix_exists=True):
    #     """

    #     """
    #     if not prefix_exists:
    #         fname = os.path.join(BASE_DIR, fname)
    #     img = imread(fname)
    #     return img

    def load_json(self, json_file):
        with open(json_file) as j:
            s = json.load(j)
        return s

    # def questions_from_image(self, img_id):
    #     questions = []
    #     return [q['question'] for q in self.q_list if q["image_id"] == img_id]

    # def load_question_batch(self, question_ids):

    #     q_ids = {q['question_id']:i for i, q in enumerate(self.q_list)}
    #     return [self.q_list[q_ids[q]]["question"] for q in q_ids]

    # def load_image_batch(self, img_ids):
    #     imgs = []
    #     for i in img_ids:
    #         id_string = "%012d" % i
    #         image_file_name = "data/train2014/" + img_name_prefix + id_string + ".jpg"
    #         output = self.image_from_file(image_file_name)
    #         imgs.append(output)
    #     return imgs
    # def load_batch_vqa(self):
    #     print("Loading annotations...")
    #     annotations = self.annotations[:self.batch_size]
    #     image_ids = []
    #     question_ids = []
    #     answers = []
    #     print("Parsing annotations...")
    #     for i in range(self.batch_size):
    #         img_id = annotations[i]['image_id']
    #         q_id = annotations[i]['question_id']
    #         image_ids.append(img_id)
    #         question_ids.append(q_id)
    #         answers.append(annotations[i]['answers'])
    #     print("Load images...")
    #     images = self.load_image_batch(image_ids)
    #     print("Load questions...")
    #     questions = self.load_question_batch(question_ids)
    #     return images, questions, answers
    # def load_imgs_batch(self, is_training):
    #     img_folder = os.path.join(BASE_DIR, "data/train2014/*.jpg")
    #     files = glob.glob(img_folder)
    #     questions = []
    #     images = []
    #     answers = []
    #     for name in files[:self.batch_size]:
    #         ids = int(name.split(".")[0].split("_")[2])
    #         img = self.image_from_file(name)
    #         qs = self.questions_from_image(ids)
    #         ans = self.answers_from_image(ids)
    #         answers.append(ans)
    #         questions.append(qs)
    #         images.append(img)

    #     return images, questions, answers

    # def load_imgs_stream(self):
    #     img_folder = os.path.join(BASE_DIR, "data/train2014/*.jpg")
    #     files = glob.glob(img_folder)
    #     for name in files:
    #         ids = int(name.split(".")[0].split("_")[2])
    #         img = self.image_from_file(name)
    #         yield (ids, img)
