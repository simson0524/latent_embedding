# data_preprocess.py

from tqdm.auto import tqdm
import numpy as np
import neurokit2 as nk
import wfdb
import h5py
import os


def get_r_peaks(lead_II, sampling_rate=125):
    """
    Get indices of R-peaks

    Args:
        signal (ndarray): signal of Lead II.
        sampling_rate (int): signal freq. Defaults to 125.

    Returns:
        Array of indices of R-peaks.
    """
    cleaned_signal = nk.ecg_clean(ecg_signal=lead_II,
                                  sampling_rate=sampling_rate)
    _, r_peaks = nk.ecg_peaks(ecg_cleaned=cleaned_signal,
                              sampling_rate=sampling_rate)
    return r_peaks['ECG_R_Peaks']


def split_and_save_all_beats_to_single_h5(signal, signal_length, r_peaks, patient_id='test'):
    """
    Save all (101, channels) shaped single beat segments into a single HDF5 file.

    Args:
        signal (ndarray): holter signal. shape (signal_length, channels)
        signal_length : total number of timesteps in signal
        r_peaks (array): R-peak indices
        patient_id (str, optional): ID string used in saved filename. Default is 'test'.
    """
    patient_id = str(patient_id)
    all_segments = []

    for idx, r_peak in tqdm(enumerate(r_peaks), total=len(r_peaks), desc=f'Extracting beats for patient {patient_id}'):
        start = max(r_peak - 50, 0)
        end = min(r_peak + 51, signal_length)
        if end - start < 101:
            continue

        curr_beat = signal[start:end, :]  # shape: (101, channels)
        all_segments.append(curr_beat)

    if not all_segments:
        print("❌ No valid segments to save.")
        return

    all_segments = np.stack(all_segments, axis=0)  # shape: (num_beats, 101, channels)

    # 저장
    save_dir = f'./single_beats'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{patient_id}_all_segments.h5')

    with h5py.File(save_path, 'w') as h5f:
        h5f.create_dataset('segments', data=all_segments)  # shape: (Num of single beats, 101, 3)
        h5f.attrs['num_segments'] = all_segments.shape[0]  # Num of single beats
        h5f.attrs['segment_length'] = 101
        h5f.attrs['num_channels'] = 3
        h5f.attrs['original_r_peak_count'] = len(r_peaks)

    print(f'✔️ Saved {all_segments.shape[0]} segments to {save_path}, shape: {all_segments.shape}')

    return save_path


# wfdb를 이용한 데이터 전처리 or denoised h5 데이터 전처리
def data_preprocess_from_raw_signal(file_path, patient_id='test'):
    """
    data preprocessing function from raw signal data(.SIG, .hea)

    Args:
        file_path : absolute path of raw signal data(without .SIG)
        patient_id : patient id for identifing. Defaults to 'test'.
    """
    record = wfdb.rdrecord(file_path)
    signal_of_record = record.p_signal
    sampling_rate_of_record = record.fs
    r_peaks = get_r_peaks(lead_II=signal_of_record[:, 2],
                          sampling_rate=sampling_rate_of_record)
    save_path = split_and_save_all_beats_to_single_h5(signal=signal_of_record,
                                                      signal_length=signal_of_record.shape[0],
                                                      r_peaks=r_peaks,
                                                      patient_id=patient_id)
    
    return r_peaks, save_path
    

def data_preprocess_from_h5(file_path, patient_id='test'):
    """
    data preprocessing function from h5 signal data.


    Args:
        file_path : absolute path of raw signal data(without .SIG)
        patient_id : patient id for identifing. Defaults to 'test'.
    """
    with h5py.File(file_path, 'r') as h5f:
        signal = h5f['denoised_ecg'][:]
        sampling_rate = h5f.attrs['fs']
    r_peaks = get_r_peaks(lead_II=signal[:, 2],
                          sampling_rate=sampling_rate)
    save_path = split_and_save_all_beats_to_single_h5(signal=signal,
                                                      signal_length=signal.shape[0],
                                                      r_peaks=r_peaks,
                                                      patient_id=patient_id)
                                            
    return r_peaks, save_path

# 아래 변수에 .SIG file path랑 해당파일 환자번호 입력
                                            
holter_signal_path = 'your_holter_signal_file_path'
patient_id = 'patient_id_of_your_data'

data_preprocess_from_raw_signal(holter_signal_path, patient_id)
# 실행결과 : 자동으로 디렉토리(save_dir)에 싱글비트단위 데이터(num beats, 101, 3) h5 생성