import pandas as pd
import os
import os.path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Model
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath('../preprocessing'))
sys.path.append(os.path.abspath('../helpfunctions'))
from mergesplits import mergesplits
from saveweights import save_model

class Wav2Vec2Classifier_librispeech(nn.Module):
    # INITIALIZE
    def __init__(self):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.feature_extractor = self.model.feature_extractor

        self.conv1 = nn.Conv1d(512, 64, kernel_size=3)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(3072, 1) 
        # (8x15872) all 10s
        # (16x3072) ddk 2s
        self.bn1 = nn.BatchNorm1d(1)
        self.bn32 = nn.BatchNorm1d(32)
        self.bn64 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

    # FORWARD PASS
    def forward(self, input_values):
        x = self.feature_extractor(input_values)
        
        x = self.conv1(x)
        x = self.bn64(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        return x

    # TRAINING AND VALIDATION
    def fit(self, k, train_dataloader, val_dataloader=None, max_epochs=10, lr=0.0001, weights_folder='./', results_folder='./'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.05)
        val_accuracy_old = 0

        all_train_loss = []
        all_val_loss = []
        all_val_acc = []
        epochs = [i for i in range(1, max_epochs+1)]
        
        for epoch in range(max_epochs):
            #print(f'--- EPOCH {epoch+1} ---')
            self.train()
            train_loss = 0.0
            
            for inputs, labels in train_dataloader:
                inputs = inputs.to(torch.float32)
                optimizer.zero_grad()
                outputs = self.forward(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss +=loss.item()
            avg_train_loss = train_loss / len(train_dataloader)
            all_train_loss.append(avg_train_loss)
                
            if val_dataloader is not None:
                self.eval()
                
                with torch.no_grad():
                    val_loss = 0.0
                    correct = 0
                    total = 0
                
                    for inputs, labels in val_dataloader:
                        inputs = inputs.to(torch.float32)
                        outputs = self.forward(inputs)
                        probabilities = torch.sigmoid(outputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        total += labels.size(0)
                        predictions = (probabilities > 0.5).float()
                        correct += torch.sum(predictions == labels).item()
                    val_accuracy = correct / total
                    avg_val_loss = val_loss / len(val_dataloader)
                    all_val_loss.append(avg_val_loss)
                    all_val_acc.append(val_accuracy)
                    save_model(folder=weights_folder, model=self, optimizer=optimizer, k=k, epoch=epoch, val=val_accuracy)
                    if val_accuracy >= val_accuracy_old:
                        val_accuracy_old = val_accuracy
        # Plotting the data
        plt.plot(all_train_loss, label='Train loss')
        plt.plot(all_val_loss, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss curves of fold {k}')
        plt.legend()
        file = os.path.join(results_folder, f'losscurves_fold_{k}.png')
        plt.savefig(file)
        plt.close()

        training_data = pd.DataFrame({'epoch': epochs, 'training_loss': all_train_loss, 'validation_loss': all_val_loss, 'validation_accuracy': all_val_acc})
        file = os.path.join(results_folder, f'fit_results_fold_{k}.csv')
        training_data.to_csv(file, index=False)
                
    # TEST FOR K FOLDS
    def test(self, test_dataloader, weights_folder, results_folder, test_id, k, max_epochs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        epochs = []
        accuracies = []
        sensitivities = []
        specificities = []
        precisions = []
        recalls = []
        cm_paths = []

        for epoch in range(1, max_epochs+1):

            weight_path = os.path.join(weights_folder, f'fold_{k}_epoch_{epoch}.pt')
            checkpoint = torch.load(weight_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.eval()
            
    
            all_predictions = []
            all_labels = []
            all_names = []
            
            for inputs, labels, names in test_dataloader:            
                inputs = inputs.to(torch.float32)
                outputs = self.forward(inputs)
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).int()
    
                predicted = predicted.tolist()
                predicted = [item for sublist in predicted for item in sublist]
                label = labels.tolist()
                label = [int(item[0]) for item in label]
    
                all_labels.append(label)
                all_predictions.append(predicted)
                all_names.append(names)

            all_predictions = [item for sublist in all_predictions for item in sublist]
            all_labels = [item for sublist in all_labels for item in sublist]
            all_names = [item for sublist in all_names for item in sublist]
    
            paths, path_preds, path_labels, ids, id_preds, id_labels = mergesplits(all_names, all_predictions, all_labels, test_id)

            classifications = pd.DataFrame({
                'path': paths,
                'predicted': path_preds,
                'label': path_labels
            })
            file = os.path.join(results_folder, f'test_classifications_fold_{k}_epoch_{epoch}.csv')
            classifications.to_csv(file, index=False)
        
            correct_path = sum(x == y for x, y in zip(path_labels, path_preds)) 
            accuracy_path = correct_path / len(path_labels)
            correct_id = sum(x == y for x, y in zip(id_labels, id_preds))
            accuracy_id = correct_id / len(id_labels)
    
            cm = confusion_matrix(path_labels, path_preds)
            TN,FP, FN, TP = cm.ravel()
            
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            epochs.append(epoch)
            accuracies.append(accuracy_path)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            precisions.append(precision)
            recalls.append(recall)
    
            f = plt.figure(figsize=(8,6))
            ax= f.add_subplot()
            sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Greens')
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title(f'Confusion Matrix epoch {epoch}')
            ax.xaxis.set_ticklabels(['HC', 'PD'])
            ax.yaxis.set_ticklabels(['HC', 'PD'])
            file = os.path.join(results_folder, f'cm_fold_{k}_epoch_{epoch}.png')
            cm_paths.append(file)
            f.savefig(file, dpi=400)
            plt.close(f)
            
        test_results = pd.DataFrame({
            'epoch': epochs,
            'accuracy': accuracies,
            'sensitivity': sensitivities,
            'specificity': specificities,
            'precision': precisions,
            'recall': recalls,
            'confusion_matrix': cm_paths 
        })
        file = os.path.join(results_folder, f'test_results_fold_{k}.csv')
        test_results.to_csv(file, index=False)

    # TESTING ONLY ONE SET OF WEIGHTS
    def test2(self, test_dataloader, weight_path, test_id, classifications_file, cm_file):
        
        checkpoint = torch.load(weight_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()
        
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []
        all_names = []
        
        for inputs, labels, names in test_dataloader:            
            inputs = inputs.to(torch.float32)
            outputs = self.forward(inputs)
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).int()

            predicted = predicted.tolist()
            predicted = [item for sublist in predicted for item in sublist]
            label = labels.tolist()
            label = [int(item[0]) for item in label]

            all_labels.append(label)
            all_predictions.append(predicted)
            all_names.append(names)

        all_predictions = [item for sublist in all_predictions for item in sublist]
        all_labels = [item for sublist in all_labels for item in sublist]
        all_names = [item for sublist in all_names for item in sublist]

        paths, path_preds, path_labels, ids, id_preds, id_labels = mergesplits(all_names, all_predictions, all_labels, test_id)

        classifications = pd.DataFrame({
            'path': paths,
            'predicted': path_preds,
            'label': path_labels
        })
        classifications.to_csv(classifications_file, index=False)
    
        correct_path = sum(x == y for x, y in zip(path_labels, path_preds)) 
        accuracy_path = correct_path / len(path_labels)
        correct_id = sum(x == y for x, y in zip(id_labels, id_preds))
        accuracy_id = correct_id / len(id_labels)

        cm = confusion_matrix(path_labels, path_preds)
        TN,FP, FN, TP = cm.ravel()
        
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        print(f'Acuracy={accuracy_path}')
        print(f'sensitivity={sensitivity:.3f}, specificity={specificity:.3f}, precision={precision:.3f}, recall={recall:.3f}')

        f = plt.figure(figsize=(8,6))
        ax= f.add_subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Greens')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix')
        ax.xaxis.set_ticklabels(['HC', 'PD'])
        ax.yaxis.set_ticklabels(['HC', 'PD'])
        f.savefig(cm_file, dpi=400)
        plt.close(f)

    # TESTING ONLY ONE SET OF WEIGHTS AND WITHOUT IDS
    def test3(self, test_dataloader, weight_path, classifications_file, cm_file):
        
        checkpoint = torch.load(weight_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()
        
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []
        all_names = []
        
        for inputs, labels, names in test_dataloader:            
            inputs = inputs.to(torch.float32)
            outputs = self.forward(inputs)
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).int()

            predicted = predicted.tolist()
            predicted = [item for sublist in predicted for item in sublist]
            label = labels.tolist()
            label = [int(item[0]) for item in label]

            all_labels.append(label)
            all_predictions.append(predicted)
            all_names.append(names)

        all_predictions = [item for sublist in all_predictions for item in sublist]
        all_labels = [item for sublist in all_labels for item in sublist]
        all_names = [item for sublist in all_names for item in sublist]

        paths, path_preds, path_labels = mergesplits2(all_names, all_predictions, all_labels)

        classifications = pd.DataFrame({
            'path': paths,
            'predicted': path_preds,
            'label': path_labels
        })
        classifications.to_csv(classifications_file, index=False)
    
        correct_path = sum(x == y for x, y in zip(path_labels, path_preds)) 
        accuracy_path = correct_path / len(path_labels)

        cm = confusion_matrix(path_labels, path_preds)
        TN,FP, FN, TP = cm.ravel()
        
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        print(f'Acuracy={accuracy_path}')
        print(f'sensitivity={sensitivity:.3f}, specificity={specificity:.3f}, precision={precision:.3f}, recall={recall:.3f}')

        f = plt.figure(figsize=(8,6))
        ax= f.add_subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Greens')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix')
        ax.xaxis.set_ticklabels(['HC', 'PD'])
        ax.yaxis.set_ticklabels(['HC', 'PD'])
        f.savefig(cm_file, dpi=400)
        plt.close(f)