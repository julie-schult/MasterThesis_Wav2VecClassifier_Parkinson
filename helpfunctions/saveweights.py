import torch
import pandas as pd
import os

BUCKET = "./weights" 

def save_model(name, model, optimizer, epoch, val):
    folder = os.path.join(BUCKET, name)
    if not os.path.exists(folder): os.makedirs(folder)
    file = os.path.join(folder, f'epoch_{epoch+1}.pt')
    #file = BUCKET + "{}/epoch_{}.pt".format(name,epoch+1)
    torch.save({
            'epoch': epoch+1,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'top1': val,
            }, file)

    print("New model, weights saved")