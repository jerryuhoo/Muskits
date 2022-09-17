# from pretty_midi import PrettyMIDI
# from visual_midi import Preset
# from visual_midi import Plotter
# import matplotlib.pyplot as plt
# from IPython.display import display, Audio
from muskit.fileio.utils import midi_to_seq
import miditoolkit
import pandas as pd
import numpy as np
import torch
import os
import soundfile as sf
import librosa
import logging
from parallel_wavegan.utils import load_model
import yaml
from muskit.svs.feats_extract.log_mel_fbank import LogMelFbank
from muskit.bin.svs_inference import SingingGenerate
import warnings

def read_label(label_str):
    line = label_str.strip().split()
    label_info = []
    for i in range(len(line) // 3):
        label_info.append(
            [line[i * 3], line[i * 3 + 1], line[i * 3 + 2]]
        )
    seq_len = len(label_info)
    sample_time = np.zeros((seq_len, 2))
    sample_label = []
    for i in range(seq_len):
        sample_time[i, 0] = np.float32(label_info[i][0])
        sample_time[i, 1] = np.float32(label_info[i][1])
        sample_label.append(label_info[i][2])
    return sample_time, sample_label


def tensorify(batch):
    for key in batch:
        batch[key] = torch.tensor(batch[key])
    return batch


# download vocoder
if not os.path.exists("/home/yyu479/svs/pretrained_models/pwg/"):
    os.system("./utils/download_from_google_drive.sh \
        https://drive.google.com/open?id=1khjnA7P-5gwmmeNsS21pgifYMnzCsIJx \
        ~/svs/pretrained_models/pwg zip")

    print("successfully finished download vocoder")
else:
    print("already downloaded pwg model")

config_path = "/home/yyu479/svs/pretrained_models/opencpop_xiaoice_nodp_model/exp/xiaoice_nodp/config.yaml"
model_path = "/home/yyu479/svs/pretrained_models/opencpop_xiaoice_nodp_model/exp/xiaoice_nodp/25epoch.pth"
vocoder_config = "/home/yyu479/svs/pretrained_models/pwg/config.yml"
vocoder_checkpoint = "/home/yyu479/svs/pretrained_models/pwg/checkpoint-250000steps.pkl"
output_path = "/home/yyu479/svs/output/"


print("start loading model!")
sing_generation = SingingGenerate(
    train_config=config_path,
    model_file=model_path,
    vocoder_config=vocoder_config,
    vocoder_checkpoint=vocoder_checkpoint
)
print("load model successfully!")


warnings.filterwarnings("ignore", category=DeprecationWarning)

egs = pd.read_csv("~/svs_demo/opencpop/egs.csv")
for index, row in egs.iterrows():
    # visualize midi
    # preset = Preset(plot_width=850)
    # plotter = Plotter(preset, plot_max_length_bar=4)
    # pm = PrettyMIDI("{}".format(row["music_score"]))
    # plotter.show_notebook(pm)

    # the format of phoneme duration information
    duration_info = open("/home/yyu479/" + row["duration"], "r", encoding="utf-8")
    duration_info = duration_info.read()
    print("duration: {}".format(duration_info))

    # load music score and conduct preprocessing
    midi_obj = miditoolkit.midi.parser.MidiFile("/home/yyu479/" + row["music_score"])
    music_score = midi_to_seq(
        midi_obj, np.int16, sing_generation.fs, 0, 1.0, "format", 0
    )

   # input information
    info = {
        "label": read_label(duration_info),
        "midi": music_score,
        "text": row["text"]
    }

    # tokenize
    batch = sing_generation.preprocess_fn(str(row["id"]), info, 1.0)
    batch = tensorify(batch)
    singing, _, _, _, _, _, _ = sing_generation(**batch)

    # let us listen to samples
    # display(Audio(singing, rate=sing_generation.fs))
    # librosa.display.waveplot(singing.cpu().numpy(), sr=sing_generation.fs)
    # plt.show()
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    sf.write(output_path + str(row["id"]) + '.wav',
             singing, sing_generation.fs, subtype='PCM_24')

# ground truth to vocoder

gt_name = "2001000001"
gt_input_path = "/home/yyu479/svs/data/Opencpop/segments/wavs/" + gt_name + ".wav"
SAMPLE_RATE = 24000
gt_data, gt_samplerate = librosa.load(gt_input_path, sr=SAMPLE_RATE)
print("sample rate:", gt_samplerate)
print("input wav len:", len(gt_data))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gt_data_tensor = torch.tensor(gt_data, device=device).float()

with open(vocoder_config) as f:
    config = yaml.load(f, Loader=yaml.Loader)
model_vocoder = load_model(vocoder_checkpoint, config)
logging.info(f"Loaded model parameters from {vocoder_checkpoint}.")
# if options.normalize_before:
# if True:
#     assert hasattr(model_vocoder, "mean"), "Feature stats are not registered."
#     assert hasattr(model_vocoder, "scale"), "Feature stats are not registered."

model_vocoder.remove_weight_norm()
model_vocoder = model_vocoder.eval().to(device)

# hop_length: 300
# n_fft: 2048
# win_length: 1200
logMelFbank = LogMelFbank(
    fs=SAMPLE_RATE,
    hop_length=300,
    n_fft=2048,
    win_length=1200
)
gt_data_tensor = torch.unsqueeze(gt_data_tensor, 0)
print("gt_data_tensor shape", gt_data_tensor.shape)
mel = logMelFbank.forward(gt_data_tensor.cpu())[0]
mel = mel.to(device)
mel = mel.squeeze()
print("mel shape", mel.shape)

wav_true = (
    model_vocoder.inference(mel, normalize_before=True).view(-1)
)
print("wav_true shape", wav_true.shape)

sf.write(
    os.path.join(output_path, "{}_true.wav".format(gt_name)),
    wav_true.cpu().detach().numpy(),
    SAMPLE_RATE,  # args.sampling_rate
    format="wav",
    subtype="PCM_24",
)
print("output gt wav after vocoder")
