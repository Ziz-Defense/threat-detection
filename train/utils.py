import subprocess, os, torch, matplotlib.pyplot as plt

def plot_learning_curves(train_loss, val_loss, val_acc, 
                         log_interval=1,filename_prefix="training_curves",
                           save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_loss) + 1)
    val_epochs = range(log_interval, len(train_loss) + 1, log_interval)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(val_epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_epochs, val_acc, label="Val Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    filepath = os.path.join(save_dir, f"{filename_prefix}.png")
    plt.savefig(filepath, dpi=400)
    print(f"Learning curves saved to {filepath}")

def get_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        commit = "unknown"
    return commit

def extract_scheduler_config(scheduler):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        return {
            "type": "ReduceLROnPlateau",
            "mode": scheduler.mode,
            "patience": scheduler.patience,
            "factor": scheduler.factor,
            "threshold": scheduler.threshold,
            "cooldown": scheduler.cooldown,
            "min_lr": scheduler.min_lrs[0] if isinstance(scheduler.min_lrs, list) else scheduler.min_lrs,
            "verbose": scheduler.verbose
        }
    else:
        return {"type": str(type(scheduler))}