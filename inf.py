import os
import torch
import librosa
import numpy as np
from scipy.io.wavfile import write
from transformers import BertTokenizer, BertModel

from models_Devc_train import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from text_cn.symbols import symbols


# ---------------------------------------------
# CONFIG
# ---------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "/home2/sanjana.hukkeri/X-E-Speech-code/promptvc_epoch_100.pth"     # <--- CHANGE THIS
config_path     = "/home2/sanjana.hukkeri/X-E-Speech-code/cross-lingual-emotional-VC/config.json"

whisper_ppg_path = "/home2/sanjana.hukkeri/X-E-Speech-code/M1_neutral_GT_largev2ppg.npy"          # <--- input PPG
ref_audio_path   = "/home2/sanjana.hukkeri/X-E-Speech-code/M1_neutral_GT.wav"           # <--- reference audio
prompt_text      = "A cheerful bright tone with high energy."  # <--- custom prompt

output_path = "/home2/sanjana.hukkeri/X-E-Speech-code/output1.wav"

# ---------------------------------------------
# LOAD HYPERPARAMETERS
# ---------------------------------------------
import json
with open(config_path, "r") as f:
    hps = json.load(f)


# ---------------------------------------------
# BERT PROMPT ENCODER (768 → 128)
# ---------------------------------------------
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model     = BertModel.from_pretrained("bert-base-uncased").to(device)
prompt_fc      = torch.nn.Linear(768, 128).to(device)

@torch.no_grad()
def get_prompt_embedding(text):
    tokens = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=40
    )

    out = bert_model(
        input_ids=tokens["input_ids"].to(device),
        attention_mask=tokens["attention_mask"].to(device)
    )

    emb = out.last_hidden_state.mean(dim=1)  # (1,768)
    emb = prompt_fc(emb)                     # (1,128)
    return emb.squeeze(0)                    # (128,)


# ---------------------------------------------
# LOAD MODEL + CHECKPOINT
# ---------------------------------------------
model = SynthesizerTrn(
    len(symbols),
    hps["data"]["filter_length"] // 2 + 1,
    hps["train"]["segment_size"] // hps["data"]["hop_length"],
    hps["model"]["inter_channels"],
    hps["model"]["hidden_channels"],
    hps["model"]["filter_channels"],
    hps["model"]["n_heads"],
    hps["model"]["n_layers"],
    hps["model"]["kernel_size"],
    hps["model"]["p_dropout"],
    hps["model"]["resblock"],
    hps["model"]["resblock_kernel_sizes"],
    hps["model"]["resblock_dilation_sizes"],
    hps["model"]["upsample_rates"],
    hps["model"]["upsample_initial_channel"],
    hps["model"]["upsample_kernel_sizes"],
    gin_channels=hps["model"]["gin_channels"],
).to(device)

ckpt = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
print("Loaded fine-tuned PromptVC checkpoint:", checkpoint_path)


# ---------------------------------------------
# LOAD REFERENCE AUDIO MEL
# ---------------------------------------------
wav_ref, _ = librosa.load(ref_audio_path, sr=hps["data"]["sampling_rate"])
wav_ref = torch.FloatTensor(wav_ref).unsqueeze(0).to(device)

ref_mel = mel_spectrogram_torch(
    wav_ref,
    hps["data"]["filter_length"],
    hps["data"]["n_mel_channels"],
    hps["data"]["sampling_rate"],
    hps["data"]["hop_length"],
    hps["data"]["win_length"],
    hps["data"]["mel_fmin"],
    hps["data"]["mel_fmax"]
).squeeze(0)

ref_mel = ref_mel.unsqueeze(0).to(device)  # (1, 80, T)


# ---------------------------------------------
# LOAD WHISPER PPG (correct shape)
# ---------------------------------------------
ppg = np.load(whisper_ppg_path)           # (T,1280)
ppg = torch.FloatTensor(ppg).transpose(0,1).unsqueeze(0).to(device)
# now = (1,1280,T)

ppg_len = torch.LongTensor([ppg.shape[-1]]).to(device)


# ---------------------------------------------
# LOAD PROMPT EMBEDDING
# ---------------------------------------------
prompt_emb = get_prompt_embedding(prompt_text).to(device)
prompt_emb = prompt_emb.unsqueeze(0)   # (1,128)


# ---------------------------------------------
# RUN INFERENCE
# ---------------------------------------------
with torch.no_grad():
    audio_out, mask, _ = model.voice_conversion_prompt(
        weo=ppg,
        weo_lengths=ppg_len,
        mel=ref_mel,
        prompt_emb=prompt_emb,
        lang=torch.LongTensor([0]).to(device),
        max_len=ref_mel.shape[-1],
    )

audio = audio_out[0, 0].cpu().numpy()
write(output_path, hps["data"]["sampling_rate"], audio)

print("✔ Saved output:", output_path)




