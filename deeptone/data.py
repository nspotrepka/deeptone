import torch
from torch.utils.data import Dataset
import torchaudio
import os

sample_rate = 44100
channels = 1
length = 1

class Audio(Dataset):
    def __init__(self, directory):
        walk = os.walk(directory)
        self.data = [z[0] + "/" + file for z in walk for file in z[2]]
        self.data = [path for path in self.data if path.endswith(".mp3")]
        self.chain = torchaudio.sox_effects.SoxEffectsChain()
        self.chain.append_effect_to_chain("rate", [str(sample_rate)])
        self.chain.append_effect_to_chain("channels", [str(channels)])
        self.chain.append_effect_to_chain("pad", ["0", str(length)])
        self.chain.append_effect_to_chain("trim", ["0", str(length)])

    def __getitem__(self, index):
        file_path = self.data[index]
        self.chain.set_input_file(file_path)
        try:
            sound, sr = self.chain.sox_build_flow_effects()
        except:
            print("found a bad audio file")
            sound = torch.zeros([channels, length * sample_rate])
            sr = sample_rate
        return sound, sr

    def __len__(self):
        return len(self.data)
