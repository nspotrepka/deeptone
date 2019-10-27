import torch
from torch.utils.data import DataLoader
import torchaudio
import deeptone

def main():
    deeptone.print_torch_version()

    directory = "fma_small"
    batch_size = 4
    num_workers = 8

    dataset = deeptone.Audio(directory)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers)
    model = deeptone.DeepTone()

    print("GPU count =", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    torchaudio.initialize_sox()

    count = 0
    for batch in loader:
        sound = batch
        sound.to(device)
        count += sound.size()[0]
        if (count % 1000 == 0):
            print("Loaded", count, "/", dataset.__len__())
    print("Done")

    torchaudio.shutdown_sox()

if __name__ == '__main__':
    main()
