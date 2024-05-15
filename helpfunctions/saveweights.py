import torch
import pandas as pd
import os

def save_model(folder, model, optimizer, epoch, val, k):
    if not os.path.exists(folder): os.makedirs(folder)
    file = os.path.join(folder, f'fold_{k}_epoch_{epoch+1}.pt')
    torch.save({
            'epoch': epoch+1,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'top1': val,
            }, file)

    print("New model, weights saved")
