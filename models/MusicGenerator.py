from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D, Reshape, Permute, RepeatVector, Concatenate, Conv3D
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop

import os
import numpy as np
from music21 import *

from music21.converter.subConverters import ConverterMusicXML
import io
from music21 import midi
from music21 import note, stream, duration, tempo
from music21 import converter
from pypianoroll import Multitrack, Track

# MusicGenerator 클래스 호출시 generator.h5파일이 같은 폴더안에 있어어ㅑ함

class MusicGenerator():
    def __init__(self):
        self.z_dim = 32
        self.n_tracks = 1
        self.n_bars = 2
        self.n_steps_per_bar = 96
        self.n_pitches = 84

        self.weight_init = RandomNormal(mean=0., stddev=0.02)  #  'he_normal' #RandomNormal(mean=0., stddev=0.02)
        self._build_generator()
        self.generator.load_weights('models/weights-g.h5')

    def conv_t(self, x, f, k, s, a, p, bn):
        x = Conv2DTranspose(
            filters=f
            , kernel_size=k
            , padding=p
            , strides=s
            , kernel_initializer=self.weight_init
        )(x)

        if bn:
            x = BatchNormalization(momentum=0.9)(x)

        if a == 'relu':
            x = Activation(a)(x)
        elif a == 'lrelu':
            x = LeakyReLU()(x)

        return x

    def TemporalNetwork(self):

        input_layer = Input(shape=(self.z_dim,), name='temporal_input')

        x = Reshape([1, 1, self.z_dim])(input_layer)
        x = self.conv_t(x, f=1024, k=(2, 1), s=(1, 1), a='relu', p='valid', bn=True)
        x = self.conv_t(x, f=self.z_dim, k=(self.n_bars - 1, 1), s=(1, 1), a='relu', p='valid', bn=True)

        output_layer = Reshape([self.n_bars, self.z_dim])(x)

        return Model(input_layer, output_layer)

    def BarGenerator(self):

        input_layer = Input(shape=(self.z_dim * 4,), name='bar_generator_input')

        x = Dense(1024)(input_layer)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = Reshape([2, 1, 512])(x)
        x = self.conv_t(x, f=512, k=(2, 1), s=(2, 1), a='relu', p='same', bn=True)  # (4, 1, 512)
        x = self.conv_t(x, f=256, k=(2, 1), s=(2, 1), a='relu', p='same', bn=True)  # (8, 1, 256)
        x = self.conv_t(x, f=256, k=(2, 1), s=(2, 1), a='relu', p='same', bn=True)  # (16, 1, 256)
        x = self.conv_t(x, f=256, k=(6, 1), s=(6, 1), a='relu', p='same', bn=True)  # (96, 1, 256)
        x = self.conv_t(x, f=256, k=(1, 7), s=(1, 7), a='relu', p='same', bn=True)  # (96, 7, 256)
        x = self.conv_t(x, f=1, k=(1, 12), s=(1, 12), a='tanh', p='same', bn=False)  # (96, 84, 1)

        output_layer = Reshape([1, self.n_steps_per_bar, self.n_pitches, 1])(x)

        return Model(input_layer, output_layer)

    def _build_generator(self):

        chords_input = Input(shape=(self.z_dim,), name='chords_input')
        style_input = Input(shape=(self.z_dim,), name='style_input')
        melody_input = Input(shape=(self.n_tracks, self.z_dim), name='melody_input') # 우리 모델에서 self.n_tracks는 항상 1
        groove_input = Input(shape=(self.n_tracks, self.z_dim), name='groove_input')

        # CHORDS -> TEMPORAL NETWORK
        self.chords_tempNetwork = self.TemporalNetwork()
        chords_over_time = self.chords_tempNetwork(chords_input)  # [n_bars, z_dim]

        # MELODY -> TEMPORAL NETWORK
        melody_over_time = [None] * self.n_tracks  # list of n_tracks [n_bars, z_dim] tensors
        self.melody_tempNetwork = [None] * self.n_tracks
        for track in range(self.n_tracks):
            self.melody_tempNetwork[track] = self.TemporalNetwork()
            melody_track = Lambda(lambda x: x[:, track, :])(melody_input)
            melody_over_time[track] = self.melody_tempNetwork[track](melody_track)

        # CREATE BAR GENERATOR FOR EACH TRACK
        self.barGen = [None] * self.n_tracks
        for track in range(self.n_tracks):
            self.barGen[track] = self.BarGenerator()

        # CREATE OUTPUT FOR EVERY TRACK AND BAR
        bars_output = [None] * self.n_bars
        for bar in range(self.n_bars):
            track_output = [None] * self.n_tracks

            c = Lambda(lambda x: x[:, bar, :], name='chords_input_bar_' + str(bar))(chords_over_time)  # [z_dim]
            s = style_input  # [z_dim]

            for track in range(self.n_tracks):
                m = Lambda(lambda x: x[:, bar, :])(melody_over_time[track])  # [z_dim]
                g = Lambda(lambda x: x[:, track, :])(groove_input)  # [z_dim]

                z_input = Concatenate(axis=1, name='total_input_bar_{}_track_{}'.format(bar, track))([c, s, m, g])

                track_output[track] = self.barGen[track](z_input)

            bars_output[bar] = track_output[0]

        generator_output = Concatenate(axis=1, name='concat_bars')(bars_output)

        self.generator = Model([chords_input, style_input, melody_input, groove_input], generator_output)

    def Generate(self):
        n = 8
        chords_noise = np.random.normal(0, 1, (n, self.z_dim))
        style_noise = np.random.normal(0, 1, (n, self.z_dim))
        melody_noise = np.random.normal(0, 1, (n, self.n_tracks, self.z_dim))
        groove_noise = np.random.normal(0, 1, (n, self.n_tracks, self.z_dim))

        score = self.generator.predict([chords_noise, style_noise, melody_noise, groove_noise])

        return score

    def notes_to_png(self, run_folder, score, filename):  # (batch_size, 4, 96, 84, 1) 의 형태
        scoreStream = self.notes_to_stream(score)
        scoreStream.write('lily.png', fp=os.path.join(run_folder, "samples/{}.png".format(filename)))

    def notes_to_midi(self, run_folder, score, filename):
        scoreStream = self.notes_to_stream(score)
        scoreStream.write('midi', fp=os.path.join(run_folder, "samples/{}.midi".format(filename)))

    def notes_to_stream(self, score):
        scoreCompressed = self.TrimScore(score, 8);
        scoreCompressed = scoreCompressed[:, :, 0:95:6, :, :]  # ( batch, 4, 16, 84, 1)
        scoreCompressed = scoreCompressed > 0.5 # 이진화 처리

        # 피치번호 37~60, 12~37으로 나눈다.
        track1 = scoreCompressed[:, :, :, 37:83, :]  # (4, 16, 23, 1) 낮은 음자리표 트랙
        track2 = scoreCompressed[:, :, :, 0:37, :]  # (4, 16, 25, 1) 높은 음자리표 트랙
        # 각각 마디와 타임스텝을 합친다. (96, 84, 1)
        track1 = track1.reshape(track1.shape[0] * track1.shape[1] * track1.shape[2], track1.shape[3])  # (96, 25)
        track2 = track2.reshape(track2.shape[0] * track2.shape[1] * track2.shape[2], track2.shape[3])  # ( 96, 23)

        scoreObject = note.Note()
        upPitch = 60
        dur = 0
        # Stream을 만든다.
        scoreStream = stream.Score()
        scoreStream.append(tempo.MetronomeMark(number=120))
        # 트랙1을 만든다.
        scoreTrack1 = stream.Part()

        # 1.타임 스텝만큼 반복한다. (96, 84)
        lastIndexes = np.array(-1)
        lengthOfTimestep = len(track1)  # 96
        for i in range(lengthOfTimestep):
            #  1.1. i번째 타임스텝의 피치들을 가져온다.
            pitches = track1[i]  # (25)
            #  1.2. 0보다 큰 피치들의 인덱스를 구한다.
            indexes = (np.where(pitches > 0))[0]  # 0보다 큰 수들의 인덱스 튜플형태로 나온다
            isEqual = np.array_equal(lastIndexes, indexes)

            if (isEqual == False or i % 16 == 0) and i > 0:  # 1.3. 이전 인덱스들과 같지 않거나 4의 배수의 타임스텝이면
                lengthOfIndexs = len(lastIndexes)  # 1.3.2 이전 인덱스 개수를 구한다.
                if lengthOfIndexs == 0:  # 1.3.3 인덱스 개수가 0개이면 쉼표를 만든다.
                    scoreObject = note.Rest()
                elif lengthOfIndexs == 1:  # 1.3.4 인덱스 개수가 1개이면 음표를 만든다.
                    scoreObject = note.Note(lastIndexes[0] + upPitch)
                else:  # 1.3.5 인덱스 개수가 2개이상이면 화음을 만든다.
                    scoreObject = chord.Chord()
                    for j in range(lengthOfIndexs):
                        scoreObject.add(note.Note(lastIndexes[j] + upPitch))
                scoreObject.duration = duration.Duration(dur)
                scoreTrack1.append(scoreObject)  # 1.3.1 만든 객체를 트랙의 추가한다.
                dur = 0

            lastIndexes = indexes
            dur += 0.25  # #  1.4. 음의길이를 센다.

        lengthOfIndexs = len(lastIndexes)  # 1.3.2 인덱스 개수를 구한다.
        if lengthOfIndexs == 0:  # 1.3.3 인덱스 개수가 0개이면 쉼표를 만든다.
            scoreObject = note.Rest()
        elif lengthOfIndexs == 1:  # 1.3.4 인덱스 개수가 1개이면 음표를 만든다.
            scoreObject = note.Note(lastIndexes[0] + upPitch)
        else:  # 1.3.5 인덱스 개수가 2개이상이면 화음을 만든다.
            scoreObject = chord.Chord()
            for j in range(lengthOfIndexs):
                scoreObject.add(note.Note(lastIndexes[j] + upPitch))
        scoreObject.duration = duration.Duration(dur)
        scoreTrack1.append(scoreObject)  # 1.3.1 만든 객체를 트랙의 추가한다.

        scoreStream.append(scoreTrack1)

        upPitch = 23
        dur = 0
        # 트랙 2를 추가한다.
        scoreTrack2 = stream.Part()
        scoreTrack2.clef = clef.BassClef()

        lastIndexes = np.array(-1)
        lengthOfTimestep = len(track2)  # 96
        for i in range(lengthOfTimestep):
            #  1.1. i번째 타임스텝의 피치들을 가져온다.
            pitches = track2[i]  # (23)
            #  1.2. 0보다 큰 피치들의 인덱스를 구한다.
            indexes = (np.where(pitches > 0))[0]  # 0보다 큰 수들의 인덱스 튜플형태로 나온다
            isEqual = np.array_equal(lastIndexes, indexes)

            if (isEqual == False or i % 16 == 0) and i > 0:  # 1.3. 이전 인덱스들과 같지 않거나 4의 배수의 타임스텝이면
                lengthOfIndexs = len(lastIndexes)  # 1.3.2 이전 인덱스 개수를 구한다.
                if lengthOfIndexs == 0:  # 1.3.3 인덱스 개수가 0개이면 쉼표를 만든다.
                    scoreObject = note.Rest()
                elif lengthOfIndexs == 1:  # 1.3.4 인덱스 개수가 1개이면 음표를 만든다.
                    scoreObject = note.Note(lastIndexes[0] + upPitch)
                else:  # 1.3.5 인덱스 개수가 2개이상이면 화음을 만든다.
                    scoreObject = chord.Chord()
                    for j in range(lengthOfIndexs):
                        scoreObject.add(note.Note(lastIndexes[j] + upPitch))
                scoreObject.duration = duration.Duration(dur)
                scoreTrack2.append(scoreObject)  # 1.3.1 만든 객체를 트랙의 추가한다.
                dur = 0

            lastIndexes = indexes
            dur += 0.25  # #  1.4. 음의길이를 센다.

        lengthOfIndexs = len(lastIndexes)  # 1.3.2 인덱스 개수를 구한다.
        if lengthOfIndexs == 0:  # 1.3.3 인덱스 개수가 0개이면 쉼표를 만든다.
            scoreObject = note.Rest()
        elif lengthOfIndexs == 1:  # 1.3.4 인덱스 개수가 1개이면 음표를 만든다.
            scoreObject = note.Note(lastIndexes[0] + upPitch)
        else:  # 1.3.5 인덱스 개수가 2개이상이면 화음을 만든다.
            scoreObject = chord.Chord()
            for j in range(lengthOfIndexs):
                scoreObject.add(note.Note(lastIndexes[j] + upPitch))
        scoreObject.duration = duration.Duration(dur)
        scoreTrack2.append(scoreObject)  # 1.3.1 만든 객체를 트랙의 추가한다.
        scoreStream.append(scoreTrack2)

        return scoreStream


    def TrimScore(self, score, leastNoteBeat):
        # score: (batchSize, 4, 96, 84 1) 형태의 배열
        # leastNoteBeat : 악보에서 나올 수 있는 음표의 최소박자
        output = np.array(score)

        batchSize = score.shape[0]
        count = score.shape[2] // leastNoteBeat;
        barCount = score.shape[1]
        pitchCount = score.shape[3]
        trackCount = score.shape[4]

        for dataNumber in range(batchSize):
            for trackNumber in range(trackCount):
                for barNumber in range(barCount):
                    for i in range(leastNoteBeat):
                        for pitchNumber in range(pitchCount):
                            output[dataNumber, barNumber, i * count:(i + 1) * count, pitchNumber, trackNumber] = score[
                                dataNumber, barNumber, i * count, pitchNumber, trackNumber]

        return output