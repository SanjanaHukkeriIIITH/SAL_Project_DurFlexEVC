import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import librosa

from transformers import BertTokenizer, BertModel
from models_Devc_train import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from text_cn.symbols import symbols


# --- CONFIG (edit paths) ---------
data_root = "/home2/sanjana.hukkeri/X-E-Speech-code/ESD_english_prompted"
hps_path = "/home2/sanjana.hukkeri/X-E-Speech-code/cross-lingual-emotional-VC/config.json"
pretrained_model_path = "/home2/sanjana.hukkeri/X-E-Speech-code/cross-lingual-emotional-VC/G_450000.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- LOAD HYPERPARAMS --------------
with open(hps_path, "r") as f:
    hps = json.load(f)


# --- PROMPT EMBEDDING MODEL -------
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()
prompt_fc = nn.Linear(768, 128).to(device)  # project BERT → 128-d

@torch.no_grad()
def get_prompt_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=50)
    outputs = bert_model(input_ids=inputs["input_ids"].to(device),
                         attention_mask=inputs["attention_mask"].to(device))
    emb = outputs.last_hidden_state.mean(dim=1)  # (1,768)
    return prompt_fc(emb).squeeze(0)  # (128,)


# --- DATASET CLASS ----------------
class PromptVCDataset(Dataset):
    def __init__(self, root, hps):
        self.items = []
        for spk in os.listdir(root):
            spk_dir = os.path.join(root, spk)
            for emo in os.listdir(spk_dir):
                emo_dir = os.path.join(spk_dir, emo)
                for fname in os.listdir(emo_dir):
                    if fname.endswith(".wav"):
                        base = fname[:-4]
                        wav = os.path.join(emo_dir, fname)
                        ppg = os.path.join(emo_dir, base + "_ppg.npy")
                        txt = os.path.join(emo_dir, base + ".txt")
                        if os.path.exists(ppg) and os.path.exists(txt):
                            self.items.append((wav, ppg, txt, spk))
        print("[Dataset] total items:", len(self.items))
        self.hps = hps

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, ppg_path, txt_path, spk = self.items[idx]
        try:
            wav, _ = librosa.load(wav_path, sr=self.hps["data"]["sampling_rate"])
            wav = torch.FloatTensor(wav).unsqueeze(0)

            mel = mel_spectrogram_torch(
                wav,
                self.hps["data"]["filter_length"],
                self.hps["data"]["n_mel_channels"],
                self.hps["data"]["sampling_rate"],
                self.hps["data"]["hop_length"],
                self.hps["data"]["win_length"],
                self.hps["data"]["mel_fmin"],
                self.hps["data"]["mel_fmax"],
            ).squeeze(0)

            ppg = torch.FloatTensor(np.load(ppg_path))

            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            prompt_emb = get_prompt_embedding(text)

            return {"wav": wav, "mel": mel, "ppg": ppg, "prompt_emb": prompt_emb, "spk": spk}

        except Exception as e:
            print("Skipping corrupt / bad file:", wav_path, e)
            # fallback — skip or choose another sample
            return None

# --- COLLATE FUNCTION -------------
def promptvc_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    batch.sort(key=lambda x: x["mel"].shape[1], reverse=True)

    wavs, mels, ppgs, mel_lens, ppg_lens, prompts = [], [], [], [], [], []
    for b in batch:
        wavs.append(b["wav"])
        mels.append(b["mel"])
        ppgs.append(b["ppg"])
        mel_lens.append(b["mel"].shape[1])
        ppg_lens.append(b["ppg"].shape[0])
        prompts.append(b["prompt_emb"])

    # MEL padding
    max_m = max(mel_lens)
    mel_dim = mels[0].shape[0]
    mel_pad = torch.zeros(len(batch), mel_dim, max_m)
    for i, m in enumerate(mels):
        mel_pad[i, :, :m.shape[1]] = m

    # PPG padding
    max_p = max(ppg_lens)
    ppg_dim = ppgs[0].shape[1]
    ppg_pad = torch.zeros(len(batch), max_p, ppg_dim)
    for i, p in enumerate(ppgs):
        ppg_pad[i, :p.shape[0], :] = p

    # WAV padding
    max_w = max(w.shape[1] for w in wavs)
    wav_pad = torch.zeros(len(batch), 1, max_w)
    for i, w in enumerate(wavs):
        wav_pad[i, :, :w.shape[1]] = w

    prompt_emb = torch.stack(prompts)

    return {
        "wav": wav_pad,
        "mel": mel_pad,
        "mel_lens": torch.LongTensor(mel_lens),
        "ppg": ppg_pad,
        "ppg_lens": torch.LongTensor(ppg_lens),
        "prompt_emb": prompt_emb
    }


# --- LOAD & INIT MODEL ---------------
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
    gin_channels=hps["model"]["gin_channels"]
).to(device)

ckpt = torch.load(pretrained_model_path, map_location="cpu")
print("Checkpoint keys:", ckpt.keys())

# try several keys
for key in ["model", "generator", "state_dict", "params"]:
    if key in ckpt:
        print(f"Loading checkpoint key: {key}")
        model.load_state_dict(ckpt[key], strict=False)
        break
else:
    raise KeyError("No valid model checkpoint key found!")

print("✅ Pretrained model loaded.")

optim = torch.optim.AdamW(model.parameters(), lr=hps["train"]["learning_rate"])
scaler = GradScaler()


# --- DATALOADER -----------------------
dataset = PromptVCDataset(data_root, hps)
loader = DataLoader(dataset, batch_size=hps["train"]["batch_size"],
                    shuffle=True, num_workers=4, collate_fn=promptvc_collate)


# --- TRAIN LOOP -----------------------
EPOCHS = hps["train"]["epochs"]   # set to 100 in your config

for epoch in range(EPOCHS):
    print(f"-- Epoch {epoch+1}/{EPOCHS} --")
    
    for batch in tqdm(loader):
        if batch is None:
            continue
        
        wav = batch["wav"].to(device)
        mel = batch["mel"].to(device)
        ppg = batch["ppg"].to(device)
        ppg_lens = batch["ppg_lens"].to(device)
        prompt_emb = batch["prompt_emb"].to(device)

        optim.zero_grad()

        with autocast():
            audio_out, mask, _ = model.voice_conversion_prompt(
                weo=ppg,
                weo_lengths=ppg_lens,
                mel=mel,
                prompt_emb=prompt_emb,
                lang=torch.zeros(len(batch), dtype=torch.long).to(device),
                max_len=mel.shape[-1]
            )

            loss = torch.mean(torch.abs(audio_out - wav[:, :, :audio_out.shape[-1]]))

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

    # ---- SAVE CHECKPOINT EVERY 20 EPOCHS ----
    if (epoch + 1) % 20 == 0:
        ckpt_out = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch + 1
        }
        out_path = f"promptvc_epoch_{epoch+1}.pth"
        torch.save(ckpt_out, out_path)
        print(f"✔ Saved checkpoint: {out_path}")

print("🎯 Training complete!")
