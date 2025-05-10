import os, random, numpy as np, librosa, json, soundfile as sf  
from scipy.io import wavfile
from typing import Union, List
from scipy.signal import butter, lfilter

def rename_files(folder_path, prefix):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    for i, f in enumerate(files):
        old_path = os.path.join(folder_path, f)
        new_name = f"{prefix}_{i:05d}.wav"
        new_path = os.path.join(folder_path,new_name)
        os.rename(old_path, new_path)
    print(f"Renaming files done")

def preprocess_non_drone(folder_path, output_folder, labels_file, target_sr=16000, target_duration=1.0):
    os.makedirs(output_folder, exist_ok=True)
    target_len = int(target_sr * target_duration)
    labels = []

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    
    for i, file in enumerate(files):
        path = os.path.join(folder_path, file)
        audio, sr = sf.read(path)

        # Convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Pad/trim
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)))

        # Normalize
        audio = audio.astype(np.float64)
        audio = audio / np.max(np.abs(audio) + 1e-7)

        filename = f"other_{i:05d}.wav"
        sf.write(os.path.join(output_folder, filename), audio, target_sr)

        # generate labels
        labels.append({
            "filename": filename,
            "label": "other",
            "count": 0
        })

    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)

    print(f" Processed {len(files)} non-drone files into {output_folder}")
    print(f" Labels saved to {labels_file}")

    return True


class augmentDataset:
    def __init__(self):
        self.folder_path = "dataset/yes_drone"
        self.noise_path = "dataset/noise"
        self.augmented_path = "dataset/augmented"
        self.labels_path = "dataset/labels"
        self.fs = 16_000
        self.audio_duration = 1.0 #s
        self.audio_length = int(self.fs*self.audio_duration)
        self.num_augment = 9_000  #non-drones sounds are 10,372 and drone sounds are 1,332
        self.one_to_three_ratio = 0.8                       

        os.makedirs(self.augmented_path, exist_ok=True)
        os.makedirs(self.labels_path, exist_ok=True)
        self.start_idx = 0

    
    def load_audio(self, file, folder_path, random_chunk=False):
        file_path = os.path.join(folder_path, file)
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if sr!= self.fs:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.fs)
        
        if (not random_chunk) and (len(audio) > self.audio_length):
                audio = audio[:self.audio_length]
        elif random_chunk and (len(audio) > self.audio_length):
            start = random.randint(0, len(audio) - self.audio_length)
            audio = audio[start:start + self.audio_length]
        else:
            audio = np.pad(audio, (0,self.audio_length - len(audio)))
        return audio.astype(np.float64)
    

    def normalize_original_data(self, labels, counter):
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(".wav")]
        for file in files:
            audio = self.load_audio(file, self.folder_path)
            normalized = audio / np.max(np.abs(audio) + 1e-7)
            
            file_name = f"drone_{counter:05d}.wav"
            sf.write(os.path.join(self.augmented_path, file_name), normalized, self.fs)

            labels.append({
                "filename": file_name,
                "label": "drone",
                "count": 1,
                "aug_type": "original_normalized"
            })

            counter += 1

        self.start_idx = counter
        return labels
            
    def mix_sounds(self, clips):
        '''
        Mix drone sound clips to mimic multiple sources
        '''
        weights = np.random.dirichlet(np.ones(len(clips)))
        mixed = sum(w*clip for w, clip in zip(weights, clips))
        return mixed/np.max(np.abs(mixed)+1e-7)
    

    def mix_noise(self,drone_audio, noise_level=0.4):
        '''
        Mix realistic outdoor noise with the drone sounds
        '''
        noisy_files = os.listdir(self.noise_path)
        rnd_file = np.random.choice(noisy_files)
        noise_audio = self.load_audio(rnd_file,self.noise_path, random_chunk=True)
        noise = noise_level * noise_audio
        mixed = drone_audio + noise 
        return mixed / np.max(np.abs(mixed) + 1e-7)

    def apply_eq_filter(self,audio, filter_type, sr,
                        low = 400, 
                        high = 2000):
        if filter_type == 'band-pass':
            b, a = butter(4, [low/(sr/2), high/(sr/2)], btype='band')
        elif filter_type == 'low-pass':
            b, a = butter(4, high/(sr/2), btype='low')
        elif filter_type == 'high-pass':
            b, a = butter(4, low/(sr/2), btype='high')
        else:
            raise ValueError("Invalid filter type!!")
        return lfilter(b,a,audio)
    
    def apply_random_gain(self, audio, min_gain=0.5, max_gain=1.5):
        gain = random.uniform(min_gain, max_gain)
        return audio*gain/np.max(np.abs(audio)+1e-7)
    
    def apply_random_dropouts(self, audio, sr, num_dropouts=2, 
                              max_len_ms=200,):
        out = audio.copy()
        max_len = int((max_len_ms/1000)*sr) #maximum samples to dropout
        for _ in range(num_dropouts):
            start = np.random.randint(0, len(out)-max_len)
            length = random.randint(10, max_len)
            out[start:start+length] = 0
        return out
    
    def apply_doppler_effect(self, audio, max_shift=2):
        stretch_rate = np.linspace(1.0, random.uniform(1.01, max_shift), len(audio))
        t_new = np.cumsum(1.0/stretch_rate)
        t_new = t_new/ t_new[-1]*(len(audio)-1)
        return np.interp(np.arange(len(audio)), t_new, audio)
    
    def apply_dynamic_compression(self,audio, threshold=0.5, ratio=4.0):
        compressed = np.where(
            np.abs(audio) < threshold,
            audio, 
            np.sign(audio)*(threshold+ (np.abs(audio)-threshold)/ratio)
        )
        return compressed / np.max(np.abs(compressed) + 1e-7)
    
    def apply_and_write(self, drone_files, labels, target_counts, counter, func, func_name=None, **kwargs):
            for i in range(target_counts):
                drone_file = np.random.choice(drone_files)
                drone_audio = self.load_audio(drone_file, self.folder_path)
                aug_audio = func(drone_audio, **kwargs)

                file_name = f"drone_{counter:05d}.wav"
                sf.write(os.path.join(self.augmented_path, file_name), aug_audio, self.fs)

                labels.append({
                    "filename": file_name,
                    "label": "drone",
                    "count": 1,
                    "aug_type": func_name or func.__name__
                })
                counter += 1
            self.start_idx = counter
            return labels

    def augment(self):
        files = os.listdir()
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(".wav")]
        two_drones_num = 1330
        three_drones_num = 1330 
        more_than_three_num = 2074
        noise_num = 1000
        filtered_num = 900
        gain_num = 800
        dropouts_num = 400
        doppler_num = 800
        compression_num = 400

        counter = self.start_idx
        labels = []

        labels = self.normalize_original_data(labels, counter=self.start_idx)
        print(f"Normalized {len(files)} original files and added to dataset.")

        def generate_drone_mixes(target_counts, 
                           labels, 
                           drone_count:Union[int, List[int]], 
                           counter):
            for i in range(target_counts):
                if isinstance(drone_count, int):
                    curr_count = drone_count
                else:
                    curr_count = np.random.randint(drone_count[0], drone_count[1]+1)
                
                ids = np.random.choice(len(files), size=curr_count, replace=True)
                clips = [self.load_audio(files[id], self.folder_path) for id in ids]
                mixed_sound = self.mix_sounds(clips)
                file_name = f"drone_{counter:05d}.wav"
                wavfile.write(os.path.join(self.augmented_path, file_name),self.fs, mixed_sound)
                counter+=1
                labels.append({
                    "filename":file_name,
                    "label": "drone", 
                    "count": curr_count
                })  
            self.start_idx = counter   
            return labels
        
        
        augmentation_configs = [
            {"func": self.mix_noise, "target_counts": noise_num, "name": "noise", "kwargs": {"noise_level": 0.4}},
            {"func": self.apply_eq_filter, "target_counts": int(filtered_num/3), "name": "bandpass", "kwargs": {"filter_type": "band-pass", "sr": self.fs}},
            {"func": self.apply_eq_filter, "target_counts": int(filtered_num/3), "name": "highpass", "kwargs": {"filter_type": "high-pass", "sr": self.fs}},
            {"func": self.apply_eq_filter, "target_counts": int(filtered_num/3), "name": "lowpass",  "kwargs": {"filter_type": "low-pass", "sr": self.fs}},
            {"func": self.apply_random_gain, "target_counts": gain_num, "name": "gain", "kwargs": {"min_gain": 0.5, "max_gain": 1.5}},
            {"func": self.apply_random_dropouts, "target_counts": dropouts_num, "name": "dropouts", "kwargs": {"sr": self.fs}},
            {"func": self.apply_doppler_effect, "target_counts": doppler_num, "name": "doppler", "kwargs": {"max_shift": 2}},
            {"func": self.apply_dynamic_compression, "target_counts": compression_num, "name": "compression", "kwargs": {"threshold": 0.5, "ratio": 4.0}},
        ]
        
        labels = generate_drone_mixes(two_drones_num, labels, drone_count= 2, counter=self.start_idx)
        labels = generate_drone_mixes(three_drones_num, labels, drone_count= 3, counter=self.start_idx)
        labels = generate_drone_mixes(more_than_three_num, labels, drone_count= [4,8], counter=self.start_idx)
        print("Drone sounds mixing completed!") 

        for config in augmentation_configs:
            labels = self.apply_and_write(
                drone_files=files,
                labels=labels,
                target_counts=config["target_counts"],
                counter=self.start_idx,
                func=config["func"],
                func_name=config["name"],
                **config["kwargs"]
            )
            print(f"{config['name']} augmentation completed. Total samples: {len(labels)}")

        labels_path = os.path.join(self.labels_path, "augmented_labels.json")
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)
        
        print(f"Saved {len(labels)} labels to {labels_path}")
            
if __name__ == '__main__':
    # ag= augmentDataset()
    # ag.augment()

    #preprocess non-drone audio
    # preprocess_non_drone(
    # folder_path="dataset/no_drone",
    # output_folder="dataset/non-drone",
    # labels_file="dataset/labels/labels_other.json"
    # )
    ...
