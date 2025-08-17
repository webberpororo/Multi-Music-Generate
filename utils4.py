import chord_recognition
import numpy as np
import miditoolkit
import os
import mir_eval
import copy

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16 # 16/8
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480
DEFAULT_VELOCITY = 80
path_cnt = 0

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

# read notes and tempo changes from midi (assume there is only one track)
def read_items(midi_path):
    global DEFAULT_FRACTION
    # print(DEFAULT_FRACTION)
    # print(os.path.basename(midi_path).split('.')[0])
    if os.path.basename(midi_path).split('.')[0] == 'drum':
        DEFAULT_FRACTION = 8
    else:
        DEFAULT_FRACTION = 16
    only_melody = False
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)

    melody_note_items = [Item(name='Note', start=note.start, end=note.end, velocity=note.velocity, pitch=note.pitch) for
                         note in midi_obj.instruments[0].notes]
    other_note_items = []
    if len(midi_obj.instruments) > 1:
        for inst in midi_obj.instruments[1:]:
            for note in inst.notes:
                other_note_items.append(
                    Item(name='Note', start=note.start, end=note.end, velocity=note.velocity, pitch=note.pitch))
    if only_melody:
        note_items = melody_note_items
    else:
        note_items = melody_note_items + other_note_items
    note_items.sort(key=lambda x: (x.start, x.pitch)) # key為排序標準(第一為note_items.start 第二為note_items.pitch)
    # print('1:', note_items)

    shift = DEFAULT_RESOLUTION * 4  # the first N:N bar
    for note_item in note_items:
        note_item.start += shift
        note_item.end += shift
    # print('2:', note_items)

    return note_items

# quantize items
def quantize_items(items, ticks=120):
    # grid
    # print(DEFAULT_FRACTION)
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items

# extract chord
def extract_chords(items):
    method = chord_recognition.MIDIChord()
    chords = method.extract(notes=items)
    output = []
    for chord in chords:
        output.append(Item(
            name='Chord',
            start=chord[0],
            end=chord[1],
            velocity=None,
            pitch=chord[2].split('/')[0]))
    return output

# group items
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)

# item to event
def item2event(groups):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(
            name='Bar',
            time=None,
            value=None,
            text='{}'.format(n_downbeat)))
        for item in groups[i][1:-1]:
            # position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            events.append(Event(
                name='Position',
                time=item.start,
                value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                text='{}'.format(item.start)))
            if item.name == 'Note':
                # velocity
                velocity_index = np.searchsorted(
                    DEFAULT_VELOCITY_BINS,
                    item.velocity,
                    side='right') - 1
                events.append(Event(
                    name='Note Velocity',
                    time=item.start,
                    value=velocity_index,
                    text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                # pitch
                events.append(Event(
                    name='Note On',
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Note Duration',
                    time=item.start,
                    value=index, # index / 0
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
            elif item.name == 'Chord':
                events.append(Event(
                    name='Chord',
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
            elif item.name == 'Tempo':
                tempo = item.pitch
                if tempo in DEFAULT_TEMPO_INTERVALS[0]:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start,
                        tempo-DEFAULT_TEMPO_INTERVALS[0].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
                    tempo_style = Event('Tempo Class', item.start, 'mid', None)
                    tempo_value = Event('Tempo Value', item.start,
                        tempo-DEFAULT_TEMPO_INTERVALS[1].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start,
                        tempo-DEFAULT_TEMPO_INTERVALS[2].start, None)
                elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 0, None)
                elif tempo > DEFAULT_TEMPO_INTERVALS[2].stop:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 59, None)
                events.append(tempo_style)
                events.append(tempo_value)
    return events

#############################################################################################
# WRITE MIDI
#############################################################################################
def word_to_event(words, word2event):
    events = []

    piano_midi = []
    drum_midi = []
    guitar_midi = []
    bass_midi = []
    # words = np.array(words)
    # print(np.array(words).shape)
    for instrument in range(len(np.array(words))):
        # print(instrument)
        # print(word)
        # event_name, event_value = word2event.get(word).split('_')
        # print(event_name)
        # events.append(Event(event_name, None, event_value, None))
        for word in words[instrument]:
            # print(word)
            # print(type(word2event[instrument]))

            word2event_item = word2event[instrument].get(word)
            print(word2event_item)
            multi_arr = word2event_item.split('+')
            for items in multi_arr:
                if 'Ep_' in items:
                    piano_item = items.lstrip('Ep_')
                    event_name, event_value = piano_item.split('_')
                    piano_midi.append(Event(event_name, None, event_value, None))
                elif 'D_' in items:
                    drum_item = items.lstrip('D_')
                    event_name, event_value = drum_item.split('_')
                    drum_midi.append(Event(event_name, None, event_value, None))
                elif 'G_' in items:
                    # print(items)
                    guitar_item = items.lstrip('G_')
                    event_name, event_value = guitar_item.split('_')
                    guitar_midi.append(Event(event_name, None, event_value, None))
                elif 'B_' in items:
                    # print(items)
                    bass_item = items.lstrip('B_')
                    event_name, event_value = bass_item.split('_')
                    bass_midi.append(Event(event_name, None, event_value, None))

    if len(piano_midi) > 0:
        events.append(piano_midi)
    if len(drum_midi) > 0:
        events.append(drum_midi)
    if len(guitar_midi) > 0:
        events.append(guitar_midi)
    if len(bass_midi) > 0:
        events.append(bass_midi)

    return events

def write_midi(words, word2event, output_path, prompt_path=None):
    event = word_to_event(words, word2event)
    # print(events)

    # get downbeat and note (no time)
    id_cnt = 0
    for events in event:
        id_cnt += 1
        # print(events)
        # is_drum = False
        if '1/8' in events[1].value:
            print('1/8')
            is_drum = True
        else:
            print('1/16')
            is_drum = False
        print('is_drum:', is_drum)
        if id_cnt >= 4:
            instrument_id = 34
        elif id_cnt == 3:
            instrument_id = 24
        else:
            instrument_id = 0
        print('instrument_id:', instrument_id)
        temp_notes = []
        temp_chords = []
        temp_tempos = []
        for i in range(len(events)-3):
            if (events[i].name == 'ar' or events[i].name == 'Bar') and i > 0: #
                temp_notes.append('Bar')
                temp_chords.append('Bar')
                temp_tempos.append('Bar')
            elif events[i].name == 'Position' and \
                events[i+1].name == 'Note Velocity' and \
                events[i+2].name == 'Note On' and \
                events[i+3].name == 'Note Duration':
                # start time and end time from position
                position = int(events[i].value.split('/')[0]) - 1
                # velocity
                index = int(events[i+1].value)
                velocity = int(DEFAULT_VELOCITY_BINS[index])
                # print('v', velocity)
                # pitch
                pitch = int(events[i+2].value)
                # duration
                index = int(events[i+3].value)
                duration = DEFAULT_DURATION_BINS[index] # DEFAULT_DURATION_BINS[index] / 0(我覺得可以用random或其他方式增加變化)
                # print('d', duration)
                # adding
                temp_notes.append([position, velocity, pitch, duration])
            elif events[i].name == 'Position' and events[i+1].name == 'Chord':
                position = int(events[i].value.split('/')[0]) - 1
                temp_chords.append([position, events[i+1].value])
            elif events[i].name == 'Position' and \
                events[i+1].name == 'Tempo Class' and \
                events[i+2].name == 'Tempo Value':
                position = int(events[i].value.split('/')[0]) - 1
                if events[i+1].value == 'slow':
                    tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(events[i+2].value)
                elif events[i+1].value == 'mid':
                    tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(events[i+2].value)
                elif events[i+1].value == 'fast':
                    tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(events[i+2].value)
                temp_tempos.append([position, tempo])
            elif events[i].name == 'None':
                print('None_event', events[i].name)
        print(temp_notes)
        print(len(temp_notes))
        print(temp_chords)
        print(temp_tempos)

        ticks_per_bar = DEFAULT_RESOLUTION * 4  # assume 4/4
        notes = []
        current_bar = 0
        for note in temp_notes:
            if note == 'Bar':
                current_bar += 1
            else:
                position, velocity, pitch, duration = note
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                # duration (end time)
                et = st + duration
                notes.append(miditoolkit.Note(DEFAULT_VELOCITY, pitch, st, et))
        # get specific time for chords
        if len(temp_chords) > 0:
            chords = []
            current_bar = 0
            for chord in temp_chords:
                if chord == 'Bar':
                    current_bar += 1
                else:
                    position, value = chord
                    # position (start time)
                    current_bar_st = current_bar * ticks_per_bar
                    current_bar_et = (current_bar + 1) * ticks_per_bar
                    flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                    st = flags[position]
                    chords.append([st, value])
        # get specific time for tempos
        tempos = []
        current_bar = 0
        for tempo in temp_tempos:
            if tempo == 'Bar':
                current_bar += 1
            else:
                position, value = tempo
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                tempos.append([int(st), value])
        # write

        if prompt_path:
            midi = miditoolkit.midi.parser.MidiFile(prompt_path)
            #
            last_time = DEFAULT_RESOLUTION * 4 * 4
            # note shift
            for note in notes:
                note.start += last_time
                note.end += last_time
            midi.instruments[0].notes.extend(notes)
            # tempo changes
            temp_tempos = []
            for tempo in midi.tempo_changes:
                if tempo.time < DEFAULT_RESOLUTION*4*4:
                    temp_tempos.append(tempo)
                else:
                    break
            for st, bpm in tempos:
                st += last_time
                temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
            midi.tempo_changes = temp_tempos
            # write chord into marker
            if len(temp_chords) > 0:
                for c in chords:
                    midi.markers.append(
                        miditoolkit.midi.containers.Marker(text=c[1], time=c[0]+last_time))
        else:
            midi = miditoolkit.midi.parser.MidiFile()
            midi.ticks_per_beat = DEFAULT_RESOLUTION
            # write instrument
            inst = miditoolkit.midi.containers.Instrument(instrument_id, is_drum=is_drum) # True/False guitar28(悶聲電)24(尼龍) bass34
            inst.notes = notes
            midi.instruments.append(inst)
            # write tempo
            tempo_changes = []
            for st, bpm in tempos:
                tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
            midi.tempo_changes = tempo_changes
            # write chord into marker
            if len(temp_chords) > 0:
                for c in chords:
                    midi.markers.append(
                        miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))

        # write
        global path_cnt
        midi.dump(output_path[path_cnt])
        path_cnt += 1
