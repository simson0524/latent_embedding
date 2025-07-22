from data_preprocess import *
from train import *
from inference import *
from dataset import load_dataloader
from model import *
from tqdm import tqdm
import pandas as pd
import torch
import h5py
import os

config = {'idle_gpus' : 1,        # 사용 가능한 GPU 개수
          'run_id': 0,            # 해당 run의 ID
          'device': 'cuda:0',     # device
          'outlier_threshold': 3  
          }

### If you have multi GPU, activate the following codes.
# config['run_id'] = int( input("Type Your [RUN ID] here ->") )
# config['device'] = input("Type your [DEVICE] here ('cuda:N' or 'cpu') ->")
# config['outlier_threshold'] = float( input("Type your [OUTLIER THRESHOLD] here ->") )

print(f'\n\n[CONFIG of current RUN]\n{config}')

### Csv file must contain 'person_id', 'file_path', 'is_psvt' columns.
# 'person_id' : Patient ID of the patient.
# 'file_path' : absolute file path of signal.
# 'is_psvt' : ground truth. 0 for negative, 1 for positive.
file_path_df = pd.read_csv("--YOUR_FILE_PATH_CSV_FILE--")
print(f'[Total] -> {len( file_path_df )}patients')


base_size = len( file_path_df ) // config['idle_gpus']
remainder = len( file_path_df ) % config['idle_gpus']
print(f"[Patient Allocation per RUN] -> {base_size}(+1) patients. remainder : {remainder}.")


splited_df = []
start_idx = 0
for i in range( config['idle_gpus'] ):
    end_idx = start_idx + base_size + (1 if i < remainder else 0)
    splited_df.append( file_path_df.iloc[start_idx:end_idx].reset_index(drop=True) )
    start_idx = end_idx
target_df = splited_df[ config['run_id'] ]
print(f'[Allocated Patient IDs] -> {target_df['person_id']}')


for i, row in tqdm(target_df.iterrows(), total=len(target_df), desc=f'Run_ID #{config["run_id"]} process'):
    # 1. Data Preprocessing
    file_path = row['file_path']
    patient_id = row['person_id']
    is_psvt = row['is_psvt']
    r_peaks, train_data_file_path = data_preprocess_from_raw_signal(file_path=file_path,
                                                                    patient_id=patient_id)
    
    # 2. Loading DataLoader
    shuffled_dataloader = load_dataloader(h5_path=train_data_file_path,
                                          batch_size=16,
                                          shuffle=True)  # dataloader for train
    not_shuffled_dataloader = load_dataloader(h5_path=train_data_file_path,
                                              batch_size=16,
                                              shuffle=False)  # dataloader for inference
    
    # 3. Train
    device = torch.device( config['device'] if torch.cuda.is_available() else 'cpu' )
    model = VAE(in_channels=3, latent_dim=4)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-6,
                                 weight_decay=0.01)
    trained_model = train(model, shuffled_dataloader, optimizer, device, epochs=20)
    os.makedirs("./VAE_weights", exist_ok=True)
    torch.save(trained_model.state_dict(), f'./VAE_weights/{patient_id}_vae.pth')

    # 4. Inference(plot 안씀 버전)
    _, _, _, std_z = extract_latents(trained_model, not_shuffled_dataloader, device)
    print(f'standardized_latent_z_shape of {patient_id} -> {std_z.shape}')
    # outlier_indices = plot_3d_latent_with_color_strips(embeddings=std_z,
    #                                                    save_dir='./plot',
    #                                                    patient_id=patient_id,
    #                                                    outlier_threshold=config['outlier_threshold'])
    # outlier_ranges = get_outlier_ranges(outlier_indices=outlier_indices,
    #                                     r_peak_indices=r_peaks)
    os.makedirs('./latent_embeddings', exist_ok=True)
    embedding_path = os.path.join('./latent_embeddings', f'{patient_id}_latent_embedding.h5')
    with h5py.File(embedding_path, 'w') as h5f:
        emb_dset = h5f.create_dataset("standardized_latent_z", data=std_z)
        emb_dset.attrs['is_psvt'] = is_psvt
        # h5f.create_dataset('outlier_indices', data=outlier_indices)
        # h5f.create_dataset('outlier_ranges', data=outlier_ranges)