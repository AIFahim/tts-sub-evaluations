from tts_scores.clvp import CLVPMetric
cv_metric = CLVPMetric(device='cuda')
score_create = cv_metric.compute_fd('/home/asif/tts_all/TTS_Evaluations/TTS_Data_for_Comparisions/align_22k_1st_sample/', '/home/asif/tts_all/TTS_Evaluations/TTS_Data_for_Comparisions/ground_truth_22k_1st_sample/')

score_collect = cv_metric.compute_fd('/home/asif/tts_all/TTS_Evaluations/TTS_Data_for_Comparisions/align_22k_collect_1st_sample/', '/home/asif/tts_all/TTS_Evaluations/TTS_Data_for_Comparisions/ground_truth_22k_1st_sample/')

print("------------- Create Align VS GT  ---------------------")
print(score_create)
print("------------- Collect Align VS GT  ---------------------")
print(score_collect)