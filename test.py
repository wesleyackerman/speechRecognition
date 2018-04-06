import soundfile as sf
import numpy as np

data, sample_rate = sf.read("librispeech/dev-clean/84/121123/84-121123-0001.flac")
print(type(data))
print(data.shape)
print(sample_rate)
sf.write("test.flac", data, int(sample_rate*.5))

