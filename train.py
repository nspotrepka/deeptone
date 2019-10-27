import torchaudio
import deeptone.data as data
import deeptone.net as net
import deeptone.setup as setup

def main():
    print('PyTorch', setup.torch_version())
    print('CUDA is available:', setup.cuda_is_available())
    print('CUDA device count:', setup.cuda_device_count())

    directory = 'fma_small'
    batch_size = 8
    num_workers = 8
    dataset = data.Audio(directory)
    loader = setup.load(dataset, batch_size, num_workers)

    device = setup.device()

    model = setup.parallel(net.Example())
    model.to(device)

    torchaudio.initialize_sox()

    count = 0
    for batch in loader:
        sound = batch
        sound.to(device)
        count = min(count + batch_size, dataset.__len__())
        print('Loaded', count, '/', dataset.__len__())
    print('Done')

    torchaudio.shutdown_sox()

if __name__ == '__main__':
    main()
