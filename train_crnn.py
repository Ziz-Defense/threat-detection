import os, torch, json, numpy as np 
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from train.dataset import DroneAudioDataset
from train.model_crnn import MFCC_CRNN
from train.utils import *
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast

def train(label_files, train_split=0.8, 
          batch_size=32, n_workers=4, epochs=500, 
          log_interval=1, lr=5e-4, model_name="mfcc-crnn",
            save_dir="models/mfcc-crnn"):
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    filename_prefix = f"model={model_name}_timestamp={timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #dataset
    dataset = DroneAudioDataset(label_files=label_files,base_path="dataset")   
    train_len = int(train_split*len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
    
    #model
    model = MFCC_CRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    scaler = GradScaler()
    best_acc = 0.0

    # start traing 
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    print(f"\n Training started on {device} for {epochs} epochs")
    for epoch in range(1, epochs+1):
        
        model.train()
        train_losses = []
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                out = model(x)
                loss = criterion(out,y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch: {epoch} | Train loss:{avg_train_loss:.4f}")
        
        
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    out_val = model(x_val)
                    val_loss = criterion(out_val, y_val)
                val_losses.append(val_loss.item())
                preds = torch.argmax(out_val, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_val.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        acc = accuracy_score(val_targets, val_preds)
        val_loss_history.append(avg_val_loss)
        val_acc_history.append(acc)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)

        if acc > best_acc:
            best_acc = acc
            #save model
            model_path = os.path.join(save_dir, f"{filename_prefix}.pt")
            torch.save(model.state_dict(), model_path)

            #save config 
            config_dict={
                "model_name": model_name,
                "epochs": epochs,
                "current_epoch":epoch,
                "batch_size": train_loader.batch_size,
                "learning_rate": lr,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "num_workers": train_loader.num_workers,
                "log_interval": log_interval, 
                "validation_accuracy":f"{acc:.4f}",
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "lr_scheduler": extract_scheduler_config(scheduler),
                "git_commit": get_git_commit()
            }

            config_path = os.path.join(save_dir, f"{filename_prefix}.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            print("saved model and config file")
    print("Training Completed!!")
    plot_learning_curves(train_loss_history, val_loss_history, 
                        val_acc_history, log_interval=log_interval,
                        save_dir="plots", filename_prefix=filename_prefix)


   

if __name__ == '__main__':

    label_files = ["labels/augmented_labels.json", "labels/labels_other.json"]  
    train(label_files)