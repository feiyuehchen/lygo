# lygop

# XPF-lyrics

This is a open-source project converting multi-language lyrics into IPA phonemes. For the original XPF project, please refer to the following links:


Check out their interactive website: [The XPF Corpus](https://cohenpr-xpf.github.io/XPF/)

The preliminary manual of the corpus can be found [here](https://cohenpr-xpf.github.io/XPF/manual/xpf_manual.pdf).

## Usage

### Setup
```python
conda create -n lygop python=3.11
conda activate lygop
git clone https://github.com/feiyuehchen/lygop
cd lygop
pip install -r requirements.txt
```

### Download Dataset
You could directly download the dataset from kaggle (https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information/data) or run the following command line:
```

mkdir ../dataset
cd dataset
#!/bin/bash
curl -L -o ./genius-song-lyrics-with-language-information.zip\
https://www.kaggle.com/api/v1/datasets/download/carlosgdcj/genius-song-lyrics-with-language-information
unzip genius-song-lyrics-with-language-information.zip 

```

The defult data structure is set as 

```
|- lygop
|- dataset
    |- song_lyrics.csv
```

