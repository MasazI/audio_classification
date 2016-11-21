#encoding: utf-8

from util import *
from features import *
import dataset

DATA_HOME = "output/data/UrbanSound8K/audio"
AUDIO_EXT = "wav"
AUDIO_NUM = 3

def disp_data(names, files):
    plot_waves(names, files)


if __name__ == '__main__':
    print("DATA_HOME: %s" % (DATA_HOME))

    waves, names = dataset.get_files("fold1")

    raw_waves = raw_sounds = load_sound_files(waves)

    disp_data(names, raw_waves)


