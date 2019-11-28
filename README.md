# DeepTone

### Setup

First, download the data:
```
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip

echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | sha1sum -c -

unzip fma_metadata.zip
unzip fma_small.zip
```

If `unzip` does not work, try using [7-Zip](https://www.7-zip.org/):
```
brew install p7zip

7z x fma_metadata.zip
7z x fma_small.zip
```

### Data Description

Data | Dimension | Description
--- | --- | ---
Audio Samples | `(8000, 2, 1323000)` | 2 channels, 30 seconds, 44.1 kHz
Genre | `(8000, 8)` | one-hot encoding of 8 genres

```
['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
```
