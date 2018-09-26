from flux.datasets.vision.celeba import CelebA
from flux.backend.data import maybe_download_and_store_google_drive

file_pair2 = {"img_align_celeba.zip":"0B7EVK8r0v71pZjFTYXZWM3FlRnM",
                     "list_attr_celeba.txt":"0B7EVK8r0v71pblRyaVFSWGxPY0U"}

maybe_download_and_store_google_drive(file_pair2, "", "")