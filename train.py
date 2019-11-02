import torchaudio
from deeptone.fma import FMA
from deeptone.net import Example
import deeptone.setup as setup

def main():
    print('PyTorch', setup.torch_version())
    print('CUDA is available:', setup.cuda_is_available())
    print('CUDA device count:', setup.cuda_device_count())

    directory = 'fma_small'
    batch_size = 8
    num_workers = 8
    dataset = FMA(directory)
    loader = setup.load(dataset, batch_size, num_workers)

    device = setup.device()

    model = setup.parallel(Example())
    model.to(device)

    torchaudio.initialize_sox()
    count = 0
    for batch in loader:
        sound, genre = batch
        sound.to(device)
        genre.to(device)
        count = min(count + batch_size, dataset.__len__())
        print('Loaded', count, '/', dataset.__len__())
    print('Done')
    torchaudio.shutdown_sox()

if __name__ == '__main__':
    main()
