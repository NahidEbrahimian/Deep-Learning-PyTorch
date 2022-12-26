import numpy as np
import torch
import telebot
import torchaudio # for speech
from model import M5

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

# Load model
label_map = np.load('label_map.npy',allow_pickle='TRUE')
num_classes = len(label_map)
model = M5(n_output = num_classes).to(device)
model.load_state_dict(torch.load("/content/drive/MyDrive/04-AudioClassification/audio_classifier_weights_best.pth"))
model.eval()

bot = telebot.TeleBot(" ")

@bot.message_handler(commands=['start'])
def start(messages):
    bot.send_message(messages.chat.id, f'welcome dear {messages.from_user.first_name} ')
    bot.send_message(messages.chat.id, f'***Audio Classification***')
    bot.send_message(messages.chat.id, f'Please send me your voice😊')

@bot.message_handler(content_types=['voice'])
def voice(message):
    audio_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(audio_info.file_path)
    src = audio_info.file_path

    with open(src, 'wb') as audio_file:
        audio_file.write(downloaded_file)

    signal, sample_rate = torchaudio.load(src)

    # preprocess
    signal = torch.mean(signal, dim=0, keepdim=True)
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
    signal = transform(signal)
    signal = signal.unsqueeze(0).to(device)

    # process
    preds = model(signal)

    # postprocess
    preds = preds.cpu().detach().numpy()
    output = np.argmax(preds)
    print(label_map[output])
    bot.reply_to(message, label_map[output])


bot.polling()
