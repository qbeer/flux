r"""
CelebA dataset formating.

"""
from flux.datasets.dataset import Dataset
from flux.backend.data import maybe_download_and_store_google_drive
from flux.util.logging import log_message
from flux.backend.globals import DATA_STORE


class CelebA(Dataset):

    def __init__(self, ):
        file_pair = {"img_align_celeba.zip":"0B7EVK8r0v71pZjFTYXZWM3FlRnM",
                     "list_attr_celeba.txt":"0B7EVK8r0v71pblRyaVFSWGxPY0U"}
        self.keys = maybe_download_and_store_google_drive(file_pair, root_key="celebA")

        # Extract each batch
        log_message('Extracting CelebA data...')
        # for i in range(1, 6):
        #     fpath = DATA_STORE['cifar-10/cifar-10-batches-py/data_batch_{}'.format(str(i))]
        #     with open(fpath, 'rb') as f:
        #         d = pickle.load(f, encoding='latin1')
        #         data = np.array(d["data"])
        #         labels = np.array(d["labels"])
        #     if i == 1:
        #         self.X_train: np.ndarray = data
        #         self.Y_train: np.ndarray = labels
        #     else:
        #         self.X_train = np.concatenate([self.X_train, data], axis=0)
        #         self.Y_train = np.concatenate([self.Y_train, labels], axis=0)

        # with open(DATA_STORE['cifar-10/cifar-10-batches-py/test_batch'], 'rb') as f:
        #     d = pickle.load(f, encoding='latin1')
        #     self.X_test: np.ndarray = np.array(d["data"])
        #     self.Y_test: np.ndarray = np.array(d["labels"])

        # # Normalize and reshape the training and test images so that they lie between 0 and 1
        # # as well as are in the right shape
        # self.X_train = np.dstack((self.X_train[:, :1024], self.X_train[:, 1024:2048],
        #                           self.X_train[:, 2048:])) / 255.
        # self.X_train = np.reshape(self.X_train, [-1, 32, 32, 3])
        # self.X_test = np.dstack((self.X_test[:, :1024], self.X_test[:, 1024:2048],
        #                          self.X_test[:, 2048:])) / 255.
        # self.X_test = np.reshape(self.X_test, [-1, 32, 32, 3])

        # if self.one_hot:
        #     self.Y_train = np.eye(10)[self.Y_train]
        #     self.Y_test = np.eye(10)[self.Y_test]
        # return


    

    @property
    def train_db(self, ):
        return

    @property
    def val_db(self, ):
        return