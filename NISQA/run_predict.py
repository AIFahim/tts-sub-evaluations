# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
from nisqa.NISQA_model import nisqaModel
import argparse
import traceback
import os
from glob import glob
import csv
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True, type=str, help='either predict_file, predict_dir, or predict_csv')
parser.add_argument('--pretrained_model', required=True, type=str, help='file name of pretrained model (must be in current working folder)')
# parser.add_argument('--deg', type=str, help='path to speech file')
parser.add_argument('--data_dir', type=str, help='folder with speech files')
parser.add_argument('--output_dir', type=str, help='folder to ouput results.csv')
parser.add_argument('--csv_file', type=str, help='file name of csv (must be in current working folder)')
parser.add_argument('--csv_deg', type=str, help='column in csv with files name/path')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for pytorchs dataloader')
parser.add_argument('--bs', type=int, default=1, help='batch size for predicting')
parser.add_argument('--ms_channel', type=int, help='audio channel in case of stereo file')

args = parser.parse_args()
args = vars(args)

# if args['mode'] == 'predict_file':
#     if args['deg'] is None:
#         raise ValueError('--deg argument with path to input file needed')
# if args['mode'] == 'predict_dir':
#     if args['data_dir'] is None:
#         raise ValueError('--data_dir argument with folder with input files needed')
# elif args['mode'] == 'predict_csv':
#     if args['csv_file'] is None:
#         raise ValueError('--csv_file argument with csv file name needed')
#     if args['csv_deg'] is None:
#         raise ValueError('--csv_deg argument with csv column name of the filenames needed')
#     if args['data_dir'] is None:
#         args['data_dir'] = ''
# else:
#         raise NotImplementedError('--mode given not available')
args['tr_bs_val'] = args['bs']
args['tr_num_workers'] = args['num_workers']
# args['deg'] = "dummy"
# print(args)



# Create and open a CSV file for writing
output_file = os.path.join(args['output_dir'], "tts_compare.csv")
with open(output_file, mode='w', newline='') as csv_file:
    fieldnames = ['deg', 'mos_pred', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred', 'model']

    # Create a CSV writer
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write the header row
    csv_writer.writeheader()

    if __name__ == "__main__":
        try:
            # Get a list of valid audio file paths
            files = glob(os.path.join("/home/asif/tts_all/TTS_Evaluations/NISQA/tts_comparision_files/", '*.wav'))
            
            # Iterate through the audio files and make predictions
            for file in files:
                args['deg'] = file  # Update the 'deg' argument with the current file path
                # Initialize the NISQA model
                nisqa = nisqaModel(args)
                
                try: 
                    nisqa.predict()
                    # Get the predictions as a dictionary
                    predictions_dict = nisqa.ds_val.df.iloc[0].to_dict()
                    # Write the predictions to the CSV file
                    csv_writer.writerow(predictions_dict)
                except:
                    pass

        except Exception as e:
            # Handle the exception
            traceback_str = traceback.format_exc()
            print("Error occurred:", str(e))

            # Save the error traceback to a text file
            with open("error_log.txt", "w") as error_file:
                error_file.write(traceback_str)


# if __name__ == "__main__":
#     try:
#         # Get a list of valid audio file paths
#         files = glob(os.path.join("/UPDS/TTS/deliverables2/create/", '*.flac'))
        
#         # Initialize an empty DataFrame to store all predictions
#         all_predictions_df = pd.DataFrame()

        

#         # Iterate through the audio files and make predictions
#         for file in files:
#             args['deg'] = file  # Update the 'deg' argument with the current file path
            
#             # Initialize the NISQA model
#             nisqa = nisqaModel(args)
#             nisqa.predict()
            
#             # Create a DataFrame with the predictions
#             predictions_df = nisqa.ds_val.df.copy()
            
#             # Append the predictions to the main DataFrame
#             all_predictions_df = pd.concat([all_predictions_df, predictions_df], ignore_index=True)

#         # Save all predictions to a single CSV file
#         output_file = os.path.join(args['output_dir'], "all_predictions.csv")
#         all_predictions_df.to_csv(output_file, index=False)

#     except Exception as e:
#         # Handle the exception
#         traceback_str = traceback.format_exc()
#         print("Error occurred:", str(e))
        
#         # Save the error traceback to a text file
#         with open("error_log.txt", "w") as error_file:
#             error_file.write(traceback_str)
































