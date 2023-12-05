# Download from hf without symlink

from huggingface_hub import snapshot_download

# Download checkpoints/ folder without cache
snapshot_download(repo_id="jbetker/tts-scores-clvp",local_dir="/home/asif/tts_all/TTS_Evaluations/CVLP_Scores_TTS/ckpt", local_dir_use_symlinks=False)