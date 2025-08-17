from model4 import PopMusicTransformer
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    # declare model
    model4 = PopMusicTransformer(
        checkpoint="D:\pythonProject\popmusictransformer_tf\REMI-finetune4",
        is_training=False)

    # generate from scratch
    model4.generate(
        n_target_bar=12,
        temperature=1.2,
        topk=500, # 取n_token的1/10
        output_path=['D:\pythonProject\popmusictransformer_tf\Output_result\From_scratch_piano4.mid',
                     'D:\pythonProject\popmusictransformer_tf\Output_result\From_scratch_drum4.mid',
                     'D:\pythonProject\popmusictransformer_tf\Output_result\From_scratch_guitar4.mid',
                     'D:\pythonProject\popmusictransformer_tf\Output_result\From_scratch_bass4.mid'],
        prompt=None)

    # # generate continuation
    # model.generate(
    #     n_target_bar=16,
    #     temperature=1.2,
    #     topk=5,
    #     output_path = './result/continuation.midi',
    #     prompt = './data/evaluation/000.midi')

    # close model
    model4.close()


if __name__ == '__main__':
    main()
