import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio

class Audio(Dataset):
    rate = 44100
    channels = 1
    length = 30

    def __init__(self, directory):
        walk = os.walk(directory)
        self.paths = [x[0] + "/" + f for x in walk for f in x[2]]
        self.paths = [p for p in self.paths if p.endswith(".mp3")]
        self.chain = torchaudio.sox_effects.SoxEffectsChain()
        self.chain.append_effect_to_chain("rate", [str(Audio.rate)])
        self.chain.append_effect_to_chain("channels", [str(Audio.channels)])
        self.chain.append_effect_to_chain("pad", ["0", str(Audio.length)])
        self.chain.append_effect_to_chain("trim", ["0", str(Audio.length)])

    def __getitem__(self, index):
        path = self.paths[index]
        self.chain.set_input_file(path)
        try:
            sound, _ = self.chain.sox_build_flow_effects()
        except:
            sound = torch.zeros([Audio.channels, Audio.length * Audio.rate])
        return sound

    def __len__(self):
        return len(self.paths)

    def loader(self, batch_size, num_workers=0):
        return DataLoader(self, batch_size, True, num_workers=num_workers)
