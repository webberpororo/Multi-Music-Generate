# import os
# os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
import tensorflow as tf
import numpy as np
import miditoolkit
import modules2
import pickle
import utils4
import time
import os
import glob
import matplotlib.pyplot as plt

# 啟用 Eager Execution
# tf.compat.v1.enable_eager_execution()
tf.random.set_seed(42)

class PopMusicTransformer(object):
    def __init__(self, checkpoint, is_training=False):
        # load dictionary
        self.dictionary_path = "D:\pythonProject\REMI-tempo-checkpoint\piano_chord4.pkl".format(checkpoint)
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        self.dictionary_path2 = "D:\pythonProject\REMI-tempo-checkpoint\drum_chord4.pkl".format(checkpoint)
        self.event2word2, self.word2event2 = pickle.load(open(self.dictionary_path2, 'rb'))
        self.dictionary_path3 = "D:\pythonProject\REMI-tempo-checkpoint\guitar_chord4.pkl".format(checkpoint)
        self.event2word3, self.word2event3 = pickle.load(open(self.dictionary_path3, 'rb'))
        self.dictionary_path4 = "D:\pythonProject\REMI-tempo-checkpoint\Bass_chord4.pkl".format(checkpoint)
        self.event2word4, self.word2event4 = pickle.load(open(self.dictionary_path4, 'rb'))
        # self.event2word_double, self.word2event_double = pickle.load(open("D:\pythonProject\REMI-tempo-checkpoint\piano_drums_dict.pkl", 'rb'))
        # model settings
        self.x_len = 16
        self.mem_len = 512
        self.n_layer = 12
        self.d_embed = 768
        self.d_model = 512
        self.dropout = 0.1
        self.n_head = 4
        self.d_head = self.d_model // self.n_head
        self.d_ff = 2048
        self.n_token = len(self.event2word) + len(self.event2word2) + len(self.event2word3) + len(self.event2word4) + 3
        self.learning_rate = 0.0002
        # load model
        self.is_training = is_training
        if self.is_training:
            self.batch_size = 4
        else:
            self.batch_size = 1
        if checkpoint is not None:
            self.checkpoint_path = '{}/model-019-0.469-1'.format(checkpoint)
            self.ck = True
        else:
            self.ck = False
            self.checkpoint_path = None
        self.load_model()

    def load_model(self):
        # 定義模型
        self.y = tf.zeros((self.batch_size, self.x_len), dtype=tf.int64)
        self.mems_i = [tf.zeros((self.mem_len, 4, self.batch_size, self.d_model)) for _ in range(self.n_layer)]
        self.model = modules2.TransformerModel(
            n_token=self.n_token,
            n_layer=self.n_layer,
            d_model=self.d_model,
            d_embed=self.d_embed,
            n_head=self.n_head,
            d_head=self.d_head,
            d_inner=self.d_ff,
            dropout=self.dropout,
            dropatt=self.dropout,
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            proj_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            mem_len=self.mem_len,
            same_length=False,
            clamp_len=-1,
            untie_r=False,
            proj_same_dim=True
        )
        # self.global_step = tf.Variable(0, trainable=True)
        # 定義優化器
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=400000,
            alpha=0.004  # 最終學習率比例 (即最小學習率為 initial_learning_rate * alpha)
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        # 加載預訓練模型
        if self.ck:
            checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            # print((self.checkpoint_path))
            checkpoint.restore(self.checkpoint_path)

    def temperature_sampling(self, logits, temperature, topk):
        prediction = []
        print(logits.shape)
        for i in range(len(logits)):
            logit = logits[i]
            # print(logit.shape)
            probs = tf.nn.softmax(logit / temperature).numpy()
            if topk == 1:
                predictions = np.argmax(probs)
                prediction.append(predictions)
            else:

                first_key_drum = list(self.word2event2.keys())[0]
                first_key_guitar = list(self.word2event3.keys())[0]
                first_key_bass = list(self.word2event4.keys())[0]

                print(first_key_drum)
                print(first_key_guitar)
                print(first_key_bass)
                piano_prob = probs[0:first_key_drum]
                drum_prob = probs[first_key_drum:first_key_guitar]
                guitar_prob = probs[first_key_guitar:first_key_bass]
                bass_prob = probs[first_key_bass:]

                piano_sorted_index = np.argsort(piano_prob)[::-1]  # 將機率索引按降冪排列，取出由大到小的排序索引
                piano_candi_index = piano_sorted_index[:topk]
                piano_candi_index = np.array(sorted(piano_candi_index))

                piano_probs = [probs[i] for i in piano_candi_index]
                # print(sorted(piano_candi_index))
                print(piano_probs)
                print(sum(piano_probs))
                piano_probs /= sum(piano_probs)
                predictions = np.random.choice(piano_candi_index, size=1, p=piano_probs)[0]
                prediction.append(predictions)

                drum_sorted_index = np.argsort(drum_prob)[::-1]  # 將機率索引按降冪排列，取出由大到小的排序索引
                drum_candi_index = drum_sorted_index[:topk]
                drum_global = drum_candi_index + first_key_drum
                drum_candi_index = np.array(sorted(drum_global))

                drum_probs = [probs[i] for i in drum_candi_index]
                # print(drum_candi_index)
                print(drum_probs)
                print(sum(drum_probs))
                drum_probs /= sum(drum_probs)
                predictions = np.random.choice(drum_candi_index, size=1, p=drum_probs)[0]
                prediction.append(predictions)

                guitar_sorted_index = np.argsort(guitar_prob)[::-1]  # 將機率索引按降冪排列，取出由大到小的排序索引
                guitar_candi_index = guitar_sorted_index[:topk]
                guitar_global = guitar_candi_index + first_key_guitar
                guitar_candi_index = np.array(sorted(guitar_global))

                guitar_probs = [probs[i] for i in guitar_candi_index]
                # print(guitar_candi_index)
                print(guitar_probs)
                print(sum(guitar_probs))
                guitar_probs /= sum(guitar_probs)
                predictions = np.random.choice(guitar_candi_index, size=1, p=guitar_probs)[0]
                prediction.append(predictions)

                bass_sorted_index = np.argsort(bass_prob)[::-1]  # 將機率索引按降冪排列，取出由大到小的排序索引
                bass_candi_index = bass_sorted_index[:topk]
                bass_global = bass_candi_index + first_key_bass
                bass_candi_index = np.array(sorted(bass_global))

                bass_probs = [probs[i] for i in bass_candi_index]
                # print(bass_candi_index)
                print(bass_probs)
                print(sum(bass_probs))
                bass_probs /= sum(bass_probs)
                predictions = np.random.choice(bass_candi_index, size=1, p=bass_probs)[0]
                prediction.append(predictions)

        return prediction

    def extract_events(self, input_path):
        note_items = utils4.read_items(input_path)
        note_items = utils4.quantize_items(note_items)
        max_time = note_items[-1].end

        # chord_items = utils.extract_chords(note_items)
        # items = chord_items + note_items

        if self.ck:
            if 'chord' in self.checkpoint_path:
                chord_items = utils4.extract_chords(note_items)
                items = chord_items + note_items
            else:
                items = note_items
        else:
            items = note_items

        groups = utils4.group_items(items, max_time)
        events = utils4.item2event(groups)
        return events

    def generate(self, n_target_bar, temperature, topk, output_path, prompt=None):
        if prompt:
            words = []
            # events = self.extract_events(prompt)
            # print(events)
            # for event in events:
            #     if event.name == 'Note Velocity':
            #         event.name = 'Note Velocity'
            #         event.value = 21
            # words = [[self.event2word_double['{}_{}'.format(e.name, e.value)] for e in events]]
            # words[0].append(self.event2word_double['Bar_None'])
        else:
            words = []
            for _ in range(self.batch_size):
                ws = [[self.event2word['Ep_Bar_None']], [self.event2word2['D_Bar_None']],
                      [self.event2word3['G_Bar_None']], [self.event2word4['B_Bar_None']]]

                # print('chord')
                # tempo_classes = [v for k, v in self.event2word.items() if 'Tempo Class' in k]
                # tempo_values = [v for k, v in self.event2word.items() if 'Tempo Value' in k]
                # chords = [v for k, v in self.event2word.items() if 'Chord' in k]
                # ws.append(self.event2word['Position_1/16'])
                # # print(chords)
                # ws.append(np.random.choice(chords))
                # ws.append(self.event2word['Position_1/16'])
                # ws.append(np.random.choice(tempo_classes))
                # ws.append(np.random.choice(tempo_values))

                if 'chord' in self.checkpoint_path:
                    print('chord')
                    # tempo_classes = [v for k, v in self.event2word_double.items() if 'Tempo Class' in k]
                    # tempo_values = [v for k, v in self.event2word_double.items() if 'Tempo Value' in k]
                    # chords = [v for k, v in self.event2word_double.items() if 'Chord' in k]
                    # ws.append(self.event2word_double['Position_1/16']) # Position_1/8 Position_1/16
                    # ws.append(np.random.choice(chords))
                    # ws.append(self.event2word_double['Position_1/16']) # Position_1/8 Position_1/16
                    # ws.append(np.random.choice(tempo_classes))
                    # ws.append(np.random.choice(tempo_values))
                else:
                    print('no chord')
                    # tempo_classes = [v for k, v in self.event2word_double.items() if 'Tempo Class' in k]
                    # tempo_values = [v for k, v in self.event2word_double.items() if 'Tempo Value' in k]
                    tempo_classes = ['slow', 'mid']
                    tempo_values = [i for i in range(60)]
                    ws[0].append(self.event2word['Ep_Position_1/16']) # Position_1/8 Position_1/16
                    ws[1].append(self.event2word2['D_Position_1/8'])
                    ws[2].append(self.event2word3['G_Position_1/16'])
                    ws[3].append(self.event2word4['B_Position_1/16'])

                    classes = np.random.choice(tempo_classes)
                    values = np.random.choice(tempo_values)
                    if values > 30: # 限制速度為slow31~60 mid1~30(61~120)
                        classes = 'slow'
                    elif values < 31:
                        classes = 'mid'
                    # classes = 'mid'
                    # values = 30
                    print(classes)
                    print(values)
                    ws[0].append(self.event2word[f'Ep_Tempo Class_{classes}'])
                    ws[0].append(self.event2word[f'Ep_Tempo Value_{values}'])
                    ws[1].append(self.event2word2[f'D_Tempo Class_{classes}'])
                    ws[1].append(self.event2word2[f'D_Tempo Value_{values}'])
                    ws[2].append(self.event2word3[f'G_Tempo Class_{classes}'])
                    ws[2].append(self.event2word3[f'G_Tempo Value_{values + 40}'])
                    ws[3].append(self.event2word4[f'B_Tempo Class_{classes}'])
                    ws[3].append(self.event2word4[f'B_Tempo Value_{values + 60}'])

                    # ws.append(np.random.choice(tempo_classes))
                    # ws.append(np.random.choice(tempo_values))
                words.append(ws)
                # print(len(ws))
        # initialize mem
        # batch_m = [tf.zeros((self.mem_len, self.batch_size, self.d_model)) for _ in range(self.n_layer)]
        # generate
        original_length = len(words[0][0])
        # print(original_length)
        initial_flag = 1
        current_generated_bar = 0
        step = 0
        while current_generated_bar < n_target_bar: #and step < 2000
            # tf.keras.backend.clear_session()
            # print('while')
            if initial_flag:
                temp_x = tf.zeros((self.batch_size, len(words[0]), original_length), dtype=tf.int32)
                for b in range(self.batch_size):
                    for z, t in enumerate(words[b][0]):
                        temp_x = tf.Variable(tf.zeros(shape=(self.batch_size, len(words[0]), len(words[b][0]))))
                        temp_x[b, 0, z].assign(t)
                        temp_x[b, 1, z].assign(words[b][1][z])
                        temp_x[b, 2, z].assign(words[b][2][z])
                        temp_x[b, 3, z].assign(words[b][3][z])
                initial_flag = 0
            else:
                temp_x = tf.Variable(tf.zeros(shape=(self.batch_size, len(words[0]), 1), dtype=tf.int32))
                for b in range(self.batch_size):
                    temp_x[b, 0, 0].assign(words[b][0][-1])
                    temp_x[b, 1, 0].assign(words[b][1][-1])
                    temp_x[b, 2, 0].assign(words[b][2][-1])
                    temp_x[b, 3, 0].assign(words[b][3][-1])
            # model (prediction)
            list_len = self.n_token
            # print('total', list_len)
            # print(len(self.word2event))
            # print(len(self.event2word))
            # print(len(self.word2event2))
            # print(len(self.event2word2))
            # print(self.n_token)

            yy = tf.zeros((self.batch_size, 4, list_len), dtype=tf.int64)
            # print('train')
            _loss, _logits, _new_mem = self.model(temp_x, yy, self.mems_i, training = False)
            # print(_logits.shape)
            _logits = tf.transpose(_logits, perm=[2, 0, 1, 3]) # [2, 0, 1, 3] [1, 0, 2, 3]
            print('logits', _logits.shape)
            # sampling
            _logit = _logits[:, :, -1, 0]
            print('logit', _logit.shape)
            word = self.temperature_sampling(
                logits=_logit,
                temperature=temperature,
                topk=topk)
            print(word)
            print(current_generated_bar)
            words[0][0].append(word[0])
            # words[0][0].append(word[12])
            words[0][1].append(word[5])
            words[0][1].append(word[13])
            words[0][2].append(word[10])
            words[0][3].append(word[-1])
            # print(word)
            if step == 5: # 250/50 (word[0] == self.event2word['Ep_Bar_None'] and word[1] == self.event2word2['D_Bar_None'])
                words[0][0].append(self.event2word['Ep_Bar_None'])
                words[0][1].append(self.event2word2['D_Bar_None'])
                words[0][2].append(self.event2word3['G_Bar_None'])
                words[0][3].append(self.event2word4['B_Bar_None'])
                current_generated_bar += 1
                step = 0
            self.mems_i = _new_mem
            step += 1

        # write
        if prompt:
            utils4.write_midi(
                words=words[0][original_length:],
                word2event=self.word2event_double,
                output_path=output_path,
                prompt_path=prompt)
        else:
            utils4.write_midi(
                words=words[0],
                word2event=[self.word2event, self.word2event2, self.word2event3, self.word2event4],
                output_path=output_path,
                prompt_path=None)
        print(len(words[0]))

    def prepare_data(self, midi_paths):
        cnt = 0
        # print(self.event2word.keys())

        final_words = []
        final_p_key = 0
        final_d_key = 0
        final_g_key = 0
        final_b_key = 0
        print(len(os.listdir(midi_paths[0])))
        for p in range(len(os.listdir(midi_paths[0]))):
            print('path', os.listdir(midi_paths[0])[p])
            all_events = []
            all_events_drum = []
            all_events_guitar = []
            all_events_bass = []
            print(midi_paths[0])
            title = midi_paths[0]
            midi_path = os.path.join(title, os.listdir(midi_paths[0])[p], "*.mid") # str(p + 1)
            midi_path = glob.glob(midi_path)
            print(midi_path)
            for path in midi_path:
                cnt += 1
                events = self.extract_events(path)
                if os.path.basename(path).split('.')[0] == 'drum':
                    print('in_drum')
                    all_events_drum.append(events)
                elif os.path.basename(path).split('.')[0] == 'guitar':
                    all_events_guitar.append(events)
                    print('in_guitar')
                elif os.path.basename(path).split('.')[0] == 'bass':
                    all_events_bass.append(events)
                    print('in_bass')
                elif os.path.basename(path).split('.')[0] == 'piano':
                    print('in_piano')
                    all_events.append(events)
                # else:
                #     print('in_piano')
                #     all_events.append(events)
            time_scale = []
            multi_start = []
            multi_end = []
            if len(all_events[0]) > 0:
                multi_start.append(all_events[0][1].time)
                multi_end.append(all_events[0][-1].time)
            if len(all_events_drum[0]) > 0:
                multi_start.append(all_events_drum[0][1].time)
                multi_end.append(all_events_drum[0][-1].time)
            if len(all_events_guitar[0]) > 0:
                multi_start.append(all_events_guitar[0][1].time)
                multi_end.append(all_events_guitar[0][-1].time)
            if len(all_events_bass[0]) > 0:
                multi_start.append(all_events_bass[0][1].time)
                multi_end.append(all_events_bass[0][-1].time)
            time_scale.append(sorted(multi_start)[0])
            time_scale.append(sorted(multi_end)[-1])
            print(time_scale)
            all_words = []
            all_words_drum = []
            all_words_guitar = []
            all_words_bass = []
            # print(all_events)
            print('piano')
            for events in all_events:
                # words = []
                # print('event', events)
                word = [[] for i in range((time_scale[-1] - time_scale[0]) // 120 + 1)]
                start = (events[1].time - time_scale[0]) // 120
                now = start
                for event in events:
                    e = '{}_{}'.format("Ep_" + event.name, event.value)
                    if e in self.event2word:
                        if 'Bar' in e:
                            # print('bar')
                            # word[now].append(self.event2word[e])
                            states1 = 0
                        else:
                            now = (event.time - events[1].time) // 120 + start
                            word[now].append(self.event2word[e])
                    else:
                        if event.name == 'Note Velocity':
                            now = (event.time - events[1].time) // 120 + start
                            word[now].append(self.event2word['Ep_Note Velocity_21'])
                        else:
                            print('Ep_ something is wrong! {}'.format(e))
                # if len(words) == 0:
                #     words.append('None')
                all_words = word
            # print(all_words)
            print('drum')
            for events in all_events_drum:
                word = [[] for i in range((time_scale[-1] - time_scale[0]) // 120 + 1)]
                start = (events[1].time - time_scale[0]) // 120
                now = start
                for event in events:
                    e = '{}_{}'.format("D_" + event.name, event.value)
                    if e in self.event2word2:
                        if 'Bar' in e:
                            # print('bar')
                            # word[now].append(self.event2word2[e])
                            states1 = 0
                        else:
                            now = (event.time - events[1].time) // 120 + start
                            word[now].append(self.event2word2[e])
                    else:
                        if event.name == 'Note Velocity':
                            now = (event.time - events[1].time) // 120 + start
                            word[now].append(self.event2word2['D_Note Velocity_21'])
                        else:
                            print('D_ something is wrong! {}'.format(e))
                # if len(words) == 0:
                #     words.append('None')
                all_words_drum = word
            print('guitar')
            for events in all_events_guitar:
                word = [[] for i in range((time_scale[-1] - time_scale[0]) // 120 + 1)]
                start = (events[1].time - time_scale[0]) // 120
                now = start
                for event in events:
                    e = '{}_{}'.format("G_" + event.name, event.value)
                    if e in self.event2word3:
                        if 'Bar' in e:
                            # print('bar')
                            # word[now].append(self.event2word[e])
                            states1 = 0
                        else:
                            now = (event.time - events[1].time) // 120 + start
                            word[now].append(self.event2word3[e])
                    else:
                        if event.name == 'Note Velocity':
                            now = (event.time - events[1].time) // 120 + start
                            word[now].append(self.event2word3['G_Note Velocity_21'])
                        else:
                            print('G_something is wrong! {}'.format(e))
                # if len(words) == 0:
                #     words.append('None')
                all_words_guitar = word
            print('bass')
            for events in all_events_bass:
                word = [[] for i in range((time_scale[-1] - time_scale[0]) // 120 + 1)]
                start = (events[1].time - time_scale[0]) // 120
                now = start
                for event in events:
                    e = '{}_{}'.format("B_" + event.name, event.value)
                    if e in self.event2word4:
                        if 'Bar' in e:
                            # print('bar')
                            # word[now].append(self.event2word[e])
                            states1 = 0
                        else:
                            now = (event.time - events[1].time) // 120 + start
                            word[now].append(self.event2word4[e])
                    else:
                        if event.name == 'Note Velocity':
                            now = (event.time - events[1].time) // 120 + start
                            word[now].append(self.event2word4['B_Note Velocity_30'])
                        else:
                            print('B_something is wrong! {}'.format(e))
                # if len(words) == 0:
                #     words.append('None')
                all_words_bass = word
            for i in range(len(all_words)):
                if len(all_words[i]) == 0:
                    all_words[i].append(np.random.randint(list(self.event2word.values())[0] + 1, list(self.event2word.values())[-1] + 1))
                if len(all_words_drum[i]) == 0:
                    all_words_drum[i].append(np.random.randint(list(self.event2word2.values())[0] + 1, list(self.event2word2.values())[-1] + 1))
                if len(all_words_guitar[i]) == 0:
                    all_words_guitar[i].append(np.random.randint(list(self.event2word3.values())[0] + 1, list(self.event2word3.values())[-1] + 1))
                if len(all_words_bass[i]) == 0:
                    all_words_bass[i].append(np.random.randint(list(self.event2word4.values())[0] + 1, list(self.event2word4.values())[-1] + 1))

            # for i in range(len(all_words)):
            #     for j in range(len(all_words[i])):
            #         if all_words[i][j] == list(self.event2word.values())[0]:
            #             all_words[i][j] = np.random.randint(list(self.event2word.values())[0], list(self.event2word.values())[-1] + 1)
            #     for j in range(len(all_words_drum[i])):
            #         if all_words_drum[i][j] == list(self.event2word2.values())[0]:
            #             all_words_drum[i][j] = np.random.randint(list(self.event2word2.values())[0], list(self.event2word2.values())[-1] + 1) # 0/2406

            # print(all_words)
            # print(all_words_drum)

            # print('event', len(all_words[0]))
            # print(len(all_words_drum[0]))
            # print(len(all_words_guitar[0]))
            # print(len(all_words_bass[0]))
            # print(all_words[0][2000])

            # for i in range(1): # 或是在最前面計算有多少資料夾
            # sort = []
            # sort.append(len(all_words[0]))
            # sort.append(len(all_words_drum[0]))
            # # sort.append(len(all_words_guitar[i]))
            # # sort.append(len(all_words_bass[i]))
            # sort = sorted(sort, reverse=True) # 計算最多音符數
            # for j in range(sort[0] - len(all_words[0])):
            #     if all_words[0][0] == 'None':
            #         break
            #     all_words[0].append(0)
            # for j in range(sort[0] - len(all_words_drum[0])):
            #     if all_words_drum[0][0] == 'None':
            #         break
            #     all_words_drum[0].append(0)
            # for i in range(sort[0] - len(all_words_guitar[0])):
            #     if all_words_guitar[0][0] == 'None':
            #         break
            #     all_words_guitar[0].append(0)
            # for i in range(sort[0] - len(all_words_bass[0])):
            #     if all_words_bass[0][0] == 'None':
            #         break
            #     all_words_bass[0].append(0)
            # print(len(all_words[0]))
            # print(len(all_words_drum[0]))

            last_key = list(self.event2word.keys())[-1]
            p_dict_key = self.event2word[last_key] + 1

            last_key_drum = list(self.event2word2.keys())[-1]
            d_dict_key = self.event2word2[last_key_drum] + 1

            last_key_guitar = list(self.event2word3.keys())[-1]
            g_dict_key = self.event2word3[last_key_guitar] + 1

            last_key_bass = list(self.event2word4.keys())[-1]
            b_dict_key = self.event2word4[last_key_bass] + 1

            # dict_key_list = sorted([p_dict_key, d_dict_key])
            # dict_key = dict_key_list[-1]

            # for val in range(0, 56):
            #     dur_str = f'Ep_Tempo Value_{val}+D_Tempo Value_{val}'
            #     dict1[last_value] = dur_str
            #     dict2[dur_str] = last_value
            #     last_value += 1
            #
            # tempo_classes = ['slow', 'mid', 'fast']
            # for cls in tempo_classes:
            #     dur_str = f'Ep_Tempo Class_{cls}+D_Tempo Class_{cls}'
            #     dict1[last_value] = dur_str
            #     dict2[dur_str] = last_value
            #     last_value += 1

            event_multi = []
            chord_event_p = []
            chord_event_d = []
            chord_event_g = []
            chord_event_b = []
            for w in range(len(all_words)):
                instrument_piano = ''
                cnt_p = 0
                for index in all_words[w]:
                    # print(index)
                    # print(self.word2event[index])
                    if cnt_p == 0:
                        instrument_piano = str(self.word2event[index])
                        cnt_p += 1
                    else:
                        instrument_piano = instrument_piano + '+' + str(self.word2event[index])
                        cnt_p += 1
                if instrument_piano not in self.event2word.keys():
                    self.event2word[instrument_piano] = p_dict_key
                    self.word2event[p_dict_key] = instrument_piano
                    p_dict_key += 1
                chord_event_p.append(self.event2word[instrument_piano])

                instrument_drum = ''
                cnt_d = 0
                for index in all_words_drum[w]:
                    # print('index', index)
                    if cnt_d == 0:
                        instrument_drum = str(self.word2event2[index])
                        cnt_d += 1
                    else:
                        instrument_drum = instrument_drum + '+' + str(self.word2event2[index])
                        cnt_d += 1
                if instrument_drum not in self.event2word2.keys():
                    self.event2word2[instrument_drum] = d_dict_key
                    self.word2event2[d_dict_key] = instrument_drum
                    d_dict_key += 1
                chord_event_d.append(self.event2word2[instrument_drum])

                instrument_guitar = ''
                cnt_g = 0
                for index in all_words_guitar[w]:
                    # print('index', index)
                    if cnt_g == 0:
                        instrument_guitar = str(self.word2event3[index])
                        cnt_g += 1
                    else:
                        instrument_guitar = instrument_guitar + '+' + str(self.word2event3[index])
                        cnt_g += 1
                if instrument_guitar not in self.event2word3.keys():
                    self.event2word3[instrument_guitar] = g_dict_key
                    self.word2event3[g_dict_key] = instrument_guitar
                    g_dict_key += 1
                chord_event_g.append(self.event2word3[instrument_guitar])

                instrument_bass = ''
                cnt_b = 0
                for index in all_words_bass[w]:
                    # print('index', index)
                    if cnt_b == 0:
                        instrument_bass = str(self.word2event4[index])
                        cnt_b += 1
                    else:
                        instrument_bass = instrument_bass + '+' + str(self.word2event4[index])
                        cnt_b += 1
                if instrument_bass not in self.event2word4.keys():
                    self.event2word4[instrument_bass] = b_dict_key
                    self.word2event4[b_dict_key] = instrument_bass
                    b_dict_key += 1
                chord_event_b.append(self.event2word4[instrument_bass])

            # print(chord_event_p)
            # print(chord_event_d)
            event_multi = np.vstack((chord_event_p, chord_event_d, chord_event_g, chord_event_b))
            print(event_multi.shape)
            final_words.append(event_multi)
            print(len(final_words[p]))
            with open("D:\pythonProject\REMI-tempo-checkpoint\piano_chord4.pkl", 'wb') as f:
                pickle.dump((self.event2word, self.word2event), f)
            with open("D:\pythonProject\REMI-tempo-checkpoint\drum_chord4.pkl", 'wb') as f:
                pickle.dump((self.event2word2, self.word2event2), f)
            with open("D:\pythonProject\REMI-tempo-checkpoint\guitar_chord4.pkl", 'wb') as f:
                pickle.dump((self.event2word3, self.word2event3), f)
            with open("D:\pythonProject\REMI-tempo-checkpoint\Bass_chord4.pkl", 'wb') as f:
                pickle.dump((self.event2word4, self.word2event4), f)
            final_p_key = p_dict_key + 1
            final_d_key = d_dict_key + 1
            final_g_key = g_dict_key + 1
            final_b_key = b_dict_key + 1
        # print(final_words)

        # event2word_double, word2event_double = pickle.load(open("D:\pythonProject\REMI-tempo-checkpoint\piano_drums_dict.pkl", 'rb'))
        # self.event2word_double = event2word_double
        # self.word2event_double = word2event_double
        # self.n_token = len(event2word_double)

        print("num_check")
        # print(list(self.event2word2.values())[0])
        print(final_p_key)
        print(final_d_key)
        print(final_g_key)
        print(final_b_key)
        last_key = list(self.event2word.keys())[0]
        p_dict_key = self.event2word[last_key]
        print(p_dict_key)
        last_key_drum = list(self.event2word2.keys())[0]
        d_dict_key = self.event2word2[last_key_drum]
        print(d_dict_key)
        last_key_guitar = list(self.event2word3.keys())[0]
        g_dict_key = self.event2word3[last_key_guitar]
        print(g_dict_key)
        last_key_bass = list(self.event2word4.keys())[0]
        b_dict_key = self.event2word4[last_key_bass]
        print(b_dict_key)

        if d_dict_key - final_p_key != 0:
            self.event2word2 = {k: v + final_p_key for k, v in self.event2word2.items()}
            self.word2event2 = {k + final_p_key: v for k, v in self.word2event2.items()}
            with open("D:\pythonProject\REMI-tempo-checkpoint\drum_chord4.pkl", 'wb') as f:
                pickle.dump((self.event2word2, self.word2event2), f)

            for seg in final_words:
                for length in range(len(seg[1])):
                    seg[1][length] = seg[1][length] + final_p_key

        if g_dict_key - final_d_key != 0:
            self.event2word3 = {k: v + final_d_key + final_p_key for k, v in self.event2word3.items()}
            self.word2event3 = {k + final_d_key + final_p_key: v for k, v in self.word2event3.items()}
            with open("D:\pythonProject\REMI-tempo-checkpoint\guitar_chord4.pkl", 'wb') as f:
                pickle.dump((self.event2word3, self.word2event3), f)

            for seg in final_words:
                for length in range(len(seg[2])):
                    seg[2][length] = seg[2][length] + final_d_key

        if b_dict_key - final_g_key != 0:
            self.event2word4 = {k: v + final_g_key + final_d_key + final_p_key for k, v in self.event2word4.items()}
            self.word2event4 = {k + final_g_key + final_d_key + final_p_key: v for k, v in self.word2event4.items()}
            with open("D:\pythonProject\REMI-tempo-checkpoint\Bass_chord4.pkl", 'wb') as f:
                pickle.dump((self.event2word4, self.word2event4), f)

            for seg in final_words:
                for length in range(len(seg[3])):
                    seg[3][length] = seg[3][length] + final_g_key

        self.group_size = 5
        segments = []
        for words in final_words:
            print(np.array(words).shape)

            # print(words)
            pairs = []
            # print(words[1])
            for i in range(0, len(words[0])-self.x_len-1, self.x_len):
                x = [words[0][i:i+self.x_len], words[1][i:i+self.x_len], words[2][i:i+self.x_len], words[3][i:i+self.x_len]]
                y = [words[0][i+1:i+self.x_len+1], words[1][i+1:i+self.x_len+1], words[2][i+1:i+self.x_len+1], words[3][i+1:i+self.x_len+1]]
                pairs.append([x, y])
            pairs = np.array(pairs)
            print('pairs', pairs.shape)
            print(len(pairs))
            for i in np.arange(0, len(pairs)-self.group_size, self.group_size*2):
                data = pairs[i:i+self.group_size]
                if len(data) == self.group_size:
                    # print('data', data)
                    segments.append(data)
        segments = np.array(segments)
        print(len(segments))
        return segments

    def finetune(self, training_data, output_checkpoint_folder):
        # shuffle
        print('finetune')
        print(self.n_token)
        index = np.arange(len(training_data))
        # np.random.shuffle(index)
        training_data = training_data[index]
        print(training_data.shape)
        print(len(training_data))
        num_batches = len(training_data) // self.batch_size
        print(num_batches)
        st = time.time()
        loss_mean = []
        for e in range(20):
            # tf.keras.backend.clear_session()
            total_loss = []
            for i in range(num_batches):
                segments = training_data[self.batch_size * i:self.batch_size * (i + 1)]
                batch_m = [tf.zeros((self.mem_len, 4, self.batch_size, self.d_model)) for _ in range(self.n_layer)]
                for j in range(self.group_size):
                    # print(segments.shape)
                    batch_x = segments[:, j, 0, :]
                    # print(batch_x.shape)
                    batch_y = segments[:, j, 1, :]
                    batch_x = tf.convert_to_tensor(batch_x, dtype=tf.int32)
                    batch_y = tf.convert_to_tensor(batch_y, dtype=tf.int32)

                    # Ensure input shape is [batch_size, seq_len]
                    # batch_x = tf.transpose(batch_x, perm=[1, 0]) if len(batch_x.shape) == 3 else batch_x
                    # batch_y = tf.transpose(batch_y, perm=[1, 0]) if len(batch_y.shape) == 3 else batch_y
                    # batch_x = tf.transpose(batch_x, perm=[1, 0])
                    # batch_y = tf.transpose(batch_y, perm=[1, 0])
                    # print(batch_x.shape)

                    with tf.GradientTape() as tape:
                        loss, logits, new_mem = self.model(batch_x, batch_y, batch_m, training=True)
                        # loss = tf.reduce_mean(loss)
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    total_norm = tf.linalg.global_norm(grads)
                    # print("Total Gradient Norm:", total_norm)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    # 如果你想手動更新 global step（不過 optimizer 也會自動跟蹤 iterations）
                    # self.global_step.assign_add(1)
                    batch_m = new_mem
                    total_loss.append(loss.numpy())
                    print('>>> Epoch: {}, Step: {}, Loss: {:.5f}, Time: {:.2f}'.format(
                        e, i, loss.numpy(), time.time() - st))
            loss_mean.append(np.mean(total_loss))
            if np.mean(total_loss) <= 0.1:
                break
        # Save model
        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        checkpoint.save('{}/model-{:03d}-{:.3f}'.format(output_checkpoint_folder, e, np.mean(total_loss)))

        plt.plot(loss_mean, label = 'loss')
        plt.legend()
        plt.show()

    def close(self):
        pass  # 在動態圖中不需要顯式關閉會話