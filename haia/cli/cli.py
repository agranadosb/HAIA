from haia.conf.env import DATA_FOLDER, BATCH_SIZE, EPOCHS, IMAGE_SIZE
from haia.constants import ANCHORS
from haia.model.trainer import VocTrainerYoloV3


def main():
    VocTrainerYoloV3(
        IMAGE_SIZE, BATCH_SIZE, EPOCHS, DATA_FOLDER, ANCHORS, training_ratio=0.8
    ).plot_model().train()
