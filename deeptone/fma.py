import csv
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

directory_metadata = 'fma_metadata'

def paths(directory_audio):
    walk = os.walk(directory_audio)
    paths = {}
    for dir in walk:
        for file in dir[2]:
            if file.endswith('.mp3'):
                i = int(file[:file.index('.mp3')])
                path = dir[0] + '/' + file
                paths[i] = path
    return paths

def genres():
    with open(directory_metadata + '/tracks.csv', newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        genre = {}
        for row in data:
            try:
                track_id = int(row[0])
                genre[track_id] = row[40]
            except ValueError:
                pass
        return genre

class FMA(Dataset):
    rate = 44100
    channels = 2
    length = 30

    def __init__(self, directory_audio):
        self.paths = paths(directory_audio)
        self.index = list(self.paths.keys())
        self.genres = {}
        for i, g in genres().items():
            if i in self.paths:
                self.genres[i] = g
        self.list_of_genres = list(set(g for g in self.genres.values()))
        self.list_of_genres.sort()
        self.chain = torchaudio.sox_effects.SoxEffectsChain()
        self.chain.append_effect_to_chain('rate', [str(FMA.rate)])
        self.chain.append_effect_to_chain('channels', [str(FMA.channels)])
        self.chain.append_effect_to_chain('pad', ['0', str(FMA.length)])
        self.chain.append_effect_to_chain('trim', ['0', str(FMA.length)])

    def __getitem__(self, i):
        path = self.paths[self.index[i]]
        self.chain.set_input_file(path)
        try:
            audio, _ = self.chain.sox_build_flow_effects()
        except RuntimeError:
            audio = torch.zeros([FMA.channels, FMA.length * FMA.rate])
        genre = self.genres[self.index[i]]
        genre_index = torch.tensor(self.list_of_genres.index(genre))
        one_hot_encoding = F.one_hot(genre_index, len(self.list_of_genres))
        return audio.float(), one_hot_encoding.float()

    def __len__(self):
        return len(self.index)
