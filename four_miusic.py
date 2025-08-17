from model4 import PopMusicTransformer
from glob import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    # declare model
    model4 = PopMusicTransformer(
        checkpoint=None,
        # "D:\pycharmprojectnew\REMI-tempo-checkpoint"
        is_training=True)
    # prepare data
    midi_paths = glob("D:\\pythonProject\\Four_music_train")  # you need to revise it
    training_data = model4.prepare_data(midi_paths=midi_paths)

    # "D:/pythonProject/music_midi2/"

    # check output checkpoint folder
    ####################################
    # if you use "REMI-tempo-chord-checkpoint" for the pre-trained checkpoint
    # please name your output folder as something with "chord"
    # for example: my-love-chord, cute-doggy-chord, ...
    # if use "REMI-tempo-checkpoint"
    # for example: my-love, cute-doggy, ...
    ####################################
    output_checkpoint_folder = 'REMI-finetune4'  # your decision
    if not os.path.exists(output_checkpoint_folder):
        os.mkdir(output_checkpoint_folder)

    # finetune
    model4.finetune(
        training_data=training_data,
        output_checkpoint_folder=output_checkpoint_folder)

    ####################################
    # after finetuning, please choose which checkpoint you want to try
    # and change the checkpoint names you choose into "model"
    # and copy the "dictionary.pkl" into the your output_checkpoint_folder
    # ***** the same as the content format in "REMI-tempo-checkpoint" *****
    # and then, you can use "main.py" to generate your own music!
    # (do not forget to revise the checkpoint path to your own in "main.py")
    ####################################

    # close
    model4.close()


if __name__ == '__main__':
    main()

