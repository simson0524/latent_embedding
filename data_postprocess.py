# data_postprocess.py

from tqdm import tqdm
import numpy as np
import h5py
import os

def extract_bag(patient_id, std_z, is_psvt, save_dir, avg_score_threshold=0.135, expected_instance_size=100, bag_scale=90):
    probability_score = np.array([0, 0.299768, 0.483466, 0.571680, 0.612574, 0.614380, 0.616124, 0.617016])

    length = std_z.shape[0]

    # Make "Score Matrix"
    bin_indices = np.clip( (std_z//0.5).astype(int), 0, len(probability_score)-1 )
    score_matrix = probability_score[bin_indices] # shape : (length, 4)
    scores = score_matrix.sum(axis=1) # shape : (length, )

    # calculate window score( shape:(expected_instance_size, ) ) using prefix sum
    window_count = length - expected_instance_size + 1
    prefix_sum = np.array([ scores[i:i+expected_instance_size].sum() for i in range(window_count) ]) # shape : (window_count, )

    # get window's start idx that the score is over than avg_score_threshold
    valid_indices = []
    i = 0
    while i < window_count:
        if prefix_sum[i] > ( avg_score_threshold*4*expected_instance_size ):
            valid_indices.append( (i, prefix_sum[i]) )
            i += (bag_scale//2)
        else:
            i += 1
    
    total_size = expected_instance_size*bag_scale
    half_total = total_size//2
    half_instance = expected_instance_size//2

    max_score_bag = []

    if valid_indices:
        max_score_bag = [ max(valid_indices, key=lambda x:x[1]) ]
    
    print(f"highest score : {max_score_bag}")

    extracted_instances_of_single_bag = []

    for i, (idx, score) in enumerate(max_score_bag):
        center = idx + half_instance
        start = center - half_total
        end = start + total_size

        print(f"Full length of 24h embeddings: {length}\nStart IDX : {start}\nEnd IDX : {end}")

        # code for exception(out of range)
        if start < 0:
            start = 0
            end = total_size
        elif end > length:
            end = length
            start = end-total_size

        print(f"Target Bag Range : {start}-{end}")

        instances_list = []

        for i in range(start, end, expected_instance_size):
            instances_list.append( std_z[i:i+expected_instance_size] )

        curr_bag = np.stack(instances_list)
        print(f"curr_bag shape(with stacked instances) : {curr_bag.shape}")

        os.makedirs(save_dir, exist_ok=True)
        embedding_path = os.path.join(save_dir, f"{patient_id}_{bag_scale}_instances_of_single_bag.h5")
        with h5py.File(embedding_path, 'w') as h5f:
            emb_dset = h5f.create_dataset("standardized_latent_z", data=curr_bag)
            emb_dset.attrs["patient_id"] = patient_id
            emb_dset.attrs["is_psvt"] = is_psvt
            emb_dset.attrs["start"] = start
            emb_dset.attrs["end"] = end

        extracted_instances_of_single_bag.append( (start, end) )

    return extracted_instances_of_single_bag, len(extracted_instances_of_single_bag)


if __name__ == "__main__":
    base_path = input("type your base path -> ")
    h5_files = [ f for f in os.listdir(base_path) if f.endswith("_latent_embedding.h5") ]
    print(f"Total h5 files count : {len(h5_files)}")

    total_bags = 0

    for file in tqdm(h5_files, total=len(h5_files), desc="Extracting bag..."):
        patient_id = file.split('_')[0]
        file_path = os.path.join(base_path, file)
        with h5py.File(file_path, 'r') as h5_files:
            embeddings = h5_files['standardized_latent_z']
            is_psvt = embeddings.attrs["is_psvt"]
            embeddings = embeddings[:]

            extracted_segments, total = extract_bag(patient_id=patient_id,
                                                    std_z=embeddings,
                                                    is_psvt=is_psvt,
                                                    save_dir=base_path+'_time_mil')
            
            total_bags += total
    
    print(f"Generated Bags : {total_bags}")
