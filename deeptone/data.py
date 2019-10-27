import torch
from torch.utils.data import Dataset
import torchaudio
import os

class Audio(Dataset):
    rate = 44100
    channels = 1
    length = 30

    def __init__(self, directory):
        walk = os.walk(directory)
        self.data = [z[0] + "/" + file for z in walk for file in z[2]]
        self.data = [path for path in self.data if path.endswith(".mp3")]
        self.chain = torchaudio.sox_effects.SoxEffectsChain()
        self.chain.append_effect_to_chain("rate", [str(Audio.rate)])
        self.chain.append_effect_to_chain("channels", [str(Audio.channels)])
        self.chain.append_effect_to_chain("pad", ["0", str(Audio.length)])
        self.chain.append_effect_to_chain("trim", ["0", str(Audio.length)])

    def __getitem__(self, index):
        file_path = self.data[index]
        self.chain.set_input_file(file_path)
        try:
            sound, _ = self.chain.sox_build_flow_effects()
        except:
            sound = torch.zeros([Audio.channels, Audio.length * Audio.rate])
        return sound

    def __len__(self):
        return len(self.data)
