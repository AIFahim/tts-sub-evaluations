import torchaudio
import torchaudio.transforms as T

# Load the original audio
original_audio, original_sample_rate = torchaudio.load('/home/asif/tts_all/NISQA_imp/TTS_Qualities/vits_22k/vits2.wav')

# Define the target sample rate (e.g., 16000 Hz)
target_sample_rate = 22050

# Initialize the resampling transform
resample_transform = T.Resample(original_sample_rate, target_sample_rate) # , normalize=True

# Resample the audio
resampled_audio = resample_transform(original_audio)

# Save the resampled audio to a new file
torchaudio.save('/home/asif/tts_all/NISQA_imp/TTS_Qualities/vits_22k/vits2.wav', resampled_audio, target_sample_rate)
