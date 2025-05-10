import os, torch, json, soundfile as sf 
from torch.utils.data import Dataset

class DroneAudioDataset(Dataset):
    def __init__(self, label_files, base_path="dataset", transform=None):
        self.items = []
        self.base_path = base_path
        self.transform = transform

        for label_file in label_files:
            with open(os.path.join(base_path, label_file), "r") as f:
                self.items.extend(json.load(f))

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        item = self.items[index]

        subfolder = "augmented" if item["label"] == "drone" else "non-drone"
        filepath = os.path.join(self.base_path, subfolder, item["filename"])

        audio,_ = sf.read(filepath)
        audio = torch.tensor(audio, dtype=torch.float32)

        count = min(item["count"], 8) #target count is 0 to 8 

        if self.transform:
            audio = self.transform(audio)

        return audio, count 
    


if __name__ == '__main__':
    ...

