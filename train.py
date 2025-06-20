import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb
import joblib

# Configuration
DATASET_PATH = "dataset.json"
MODELS_DIR = "saved_models_with_xgboost"
BATCH_SIZE = 64
EPOCHS = 50  
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8 
INPUT_SIZE = 1000  
HIDDEN_SIZE = 128

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


class FingerprintClassifier(nn.Module):
    """Basic neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(FingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After two 2x pooling operations
        self.fc_input_size = conv_output_size * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
class ComplexFingerprintClassifier(nn.Module):
    """A more complex neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexFingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After three 2x pooling operations
        self.fc_input_size = conv_output_size * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size*2)
        self.bn4 = nn.BatchNorm1d(hidden_size*2)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x



def train(model, train_loader, test_loader, criterion, optimizer, epochs, model_save_path):
    """Train a PyTorch model and evaluate on the test set.
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        model_save_path: Path to save the best model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(traces)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * traces.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Evaluation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for traces, labels in test_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs = model(traces)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * traces.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_accuracy = correct / total
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_accuracy)
        
        # Print status
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, '
              f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}')
        
        # Save the best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with accuracy: {best_accuracy:.4f}')
    
    return best_accuracy



def evaluate(model, test_loader, website_names):
    """Evaluate a PyTorch model on the test set and show classification report with website names.
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for testing data
        website_names: List of website names for classification report
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for traces, labels in test_loader:
            traces, labels = traces.to(device), labels.to(device)
            outputs = model(traces)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print classification report with website names instead of indices
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=website_names,
        zero_division=1
    ))
    
    return all_preds, all_labels


def main():
    # Load raw JSON
    with open(DATASET_PATH) as f:
        raw = json.load(f)
    # Determine format (single dict vs list of dicts)
    if isinstance(raw, dict) and 'trace_data' in raw:
        records = [raw]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("Unsupported dataset format")

    # Extract traces and sites
    traces = [r['trace_data'] for r in records]
    # print(traces[0])
    sites = [r['website'] for r in records]

    # Label mapping
    unique_sites = sorted(set(sites))
    # print(unique_sites)
    label_map = {s: i for i, s in enumerate(unique_sites)}
    # print(label_map)
    labels = [label_map[s] for s in sites]
    # print(labels)
    # Prepare X, y
    X = np.zeros((len(traces), INPUT_SIZE), dtype=np.float32)
    for i, t in enumerate(traces):
        arr = np.array(t, dtype=np.float32)
        length = min(len(arr), INPUT_SIZE)
        X[i, :length] = arr[:length]
    y = np.array(labels, dtype=np.int64)

    # Dataset & split
    class FingerprintDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
        def __len__(self): return len(self.y)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    dataset = FingerprintDataset(X, y)
    sss = StratifiedShuffleSplit(1, test_size=1-TRAIN_SPLIT, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    # print(train_idx[0])
    # print(test_idx[0])
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE)
    # print(len(train_loader.dataset))
    # print(len(test_loader.dataset))
    # Train & evaluate
    models = [
        (FingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, len(unique_sites)), 'simple'),
        (ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, len(unique_sites)), 'complex')
    ]
    for model, name in models:
        print(f"\nTraining {name} model...")
        save_path = os.path.join(MODELS_DIR, f"{name}_model.pth")
        # If a saved model exists, load its parameters before (re-)training
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            print(f"Loaded existing weights for {name} model from {save_path}")
        train(
            model,
            train_loader,
            test_loader,
            nn.CrossEntropyLoss(),
            optim.Adam(model.parameters(), lr=LEARNING_RATE),
            EPOCHS,
            save_path
        )
        print(f"\nEvaluating {name} model")
        model.load_state_dict(torch.load(save_path))
        evaluate(model, test_loader, unique_sites)
    


    # --- BEGIN XGBoost ADDED PART (fixed) ---
    print("\nTraining xgboost model with eval logging...")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        n_estimators=100,
        learning_rate=0.3,
        eval_metric='mlogloss'
    )

    # Provide both train & test for per-round logging
    eval_sets = [
        (X[train_idx], y[train_idx]),  # will be shown as validation_0
        (X[test_idx],  y[test_idx])    # will be shown as validation_1
    ]

    xgb_model.fit(
        X[train_idx], 
        y[train_idx],
        eval_set=eval_sets,
        verbose=True
    )

    # Final classification report
    xgb_preds = xgb_model.predict(X[test_idx])
    print("\nFinal XGBoost Classification Report:")
    print(classification_report(
        y[test_idx],
        xgb_preds,
        target_names=unique_sites,
        zero_division=1
    ))

    # Final train accuracy
    train_preds = xgb_model.predict(X[train_idx])
    train_acc = (train_preds == y[train_idx]).mean()
    print(f"Final XGBoost Train Acc: {train_acc:.4f}")

    # --- ADDED: Final test accuracy ---
    test_preds = xgb_model.predict(X[test_idx])
    test_acc = (test_preds == y[test_idx]).mean()
    print(f"Final XGBoost Test Acc: {test_acc:.4f}")

    # Save the model
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    # --- END XGBoost ADDED PART ---


if __name__ == "__main__":
    main()




# import os
# import json
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, Subset
# from sklearn.metrics import classification_report
# from sklearn.model_selection import StratifiedShuffleSplit

# # Configuration
# DATASET_PATH = "dataset.json"
# MODELS_DIR = "saved_models"
# BATCH_SIZE = 64
# EPOCHS = 50  
# LEARNING_RATE = 1e-4
# TRAIN_SPLIT = 0.8 
# INPUT_SIZE = 1000  
# HIDDEN_SIZE = 128

# # Ensure models directory exists
# os.makedirs(MODELS_DIR, exist_ok=True)


# class FingerprintClassifier(nn.Module):
#     """Basic neural network model for website fingerprinting classification."""
    
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(FingerprintClassifier, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2)
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
#         conv_output_size = input_size // 8  # After two 2x pooling
#         self.fc_input_size = conv_output_size * 64
#         self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = self.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = x.view(-1, self.fc_input_size)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         return self.fc2(x)
        
# class ComplexFingerprintClassifier(nn.Module):
#     """A more complex neural network model for website fingerprinting classification."""
    
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(ComplexFingerprintClassifier, self).__init__()
#         self.conv1 = nn.Conv1d(1, 32, 5, padding=2)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.pool1 = nn.MaxPool1d(2)
#         self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.pool2 = nn.MaxPool1d(2)
#         self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.pool3 = nn.MaxPool1d(2)
#         conv_output_size = input_size // 8  # After three 2x pooling
#         self.fc_input_size = conv_output_size * 128
#         self.fc1 = nn.Linear(self.fc_input_size, hidden_size*2)
#         self.bn4 = nn.BatchNorm1d(hidden_size*2)
#         self.dropout1 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(hidden_size*2, hidden_size)
#         self.bn5 = nn.BatchNorm1d(hidden_size)
#         self.dropout2 = nn.Dropout(0.3)
#         self.fc3 = nn.Linear(hidden_size, num_classes)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.pool1(x)
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.pool2(x)
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.pool3(x)
#         x = x.view(-1, self.fc_input_size)
#         x = self.relu(self.bn4(self.fc1(x)))
#         x = self.dropout1(x)
#         x = self.relu(self.bn5(self.fc2(x)))
#         x = self.dropout2(x)
#         return self.fc3(x)


# def train(model, train_loader, test_loader, criterion, optimizer, epochs, model_save_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     best_accuracy = 0.0
#     for epoch in range(epochs):
#         model.train()
#         running_loss = correct = total = 0
#         for traces, labels in train_loader:
#             traces, labels = traces.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(traces)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * traces.size(0)
#             _, preds = torch.max(outputs, 1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#         train_loss = running_loss / total
#         train_acc = correct / total

#         model.eval()
#         running_loss = correct = total = 0
#         with torch.no_grad():
#             for traces, labels in test_loader:
#                 traces, labels = traces.to(device), labels.to(device)
#                 outputs = model(traces)
#                 loss = criterion(outputs, labels)
#                 running_loss += loss.item() * traces.size(0)
#                 _, preds = torch.max(outputs, 1)
#                 correct += (preds == labels).sum().item()
#                 total += labels.size(0)
#         test_acc = correct / total
#         print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
#         if test_acc > best_accuracy:
#             best_accuracy = test_acc
#             torch.save(model.state_dict(), model_save_path)
#     return best_accuracy


# def evaluate(model, test_loader, labels):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device).eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for traces, lab in test_loader:
#             traces, lab = traces.to(device), lab.to(device)
#             outputs = model(traces)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(lab.cpu().numpy())
#     print("\n" + classification_report(all_labels, all_preds, target_names=labels, zero_division=1))


# def main():
#     # Load raw JSON
#     with open(DATASET_PATH) as f:
#         raw = json.load(f)
#     # Determine format (single dict vs list of dicts)
#     if isinstance(raw, dict) and 'trace_data' in raw:
#         records = [raw]
#     elif isinstance(raw, list):
#         records = raw
#     else:
#         raise ValueError("Unsupported dataset format")

#     # Extract traces and sites
#     traces = [r['trace_data'] for r in records]
#     # print(traces[0])
#     sites = [r['website'] for r in records]

#     # Label mapping
#     unique_sites = sorted(set(sites))
#     # print(unique_sites)
#     label_map = {s: i for i, s in enumerate(unique_sites)}
#     # print(label_map)
#     labels = [label_map[s] for s in sites]
#     # print(labels)
#     # Prepare X, y
#     X = np.zeros((len(traces), INPUT_SIZE), dtype=np.float32)
#     for i, t in enumerate(traces):
#         arr = np.array(t, dtype=np.float32)
#         length = min(len(arr), INPUT_SIZE)
#         X[i, :length] = arr[:length]
#     y = np.array(labels, dtype=np.int64)

#     # Dataset & split
#     class FingerprintDataset(Dataset):
#         def __init__(self, X, y):
#             self.X = torch.from_numpy(X)
#             self.y = torch.from_numpy(y)
#         def __len__(self): return len(self.y)
#         def __getitem__(self, idx): return self.X[idx], self.y[idx]

#     dataset = FingerprintDataset(X, y)
#     sss = StratifiedShuffleSplit(1, test_size=1-TRAIN_SPLIT, random_state=42)
#     train_idx, test_idx = next(sss.split(X, y))
#     # print(train_idx[0])
#     # print(test_idx[0])
#     train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE)
#     # print(len(train_loader.dataset))
#     # print(len(test_loader.dataset))
#     # Train & evaluate
#     models = [
#         (FingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, len(unique_sites)), 'simple'),
#         (ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, len(unique_sites)), 'complex')
#     ]
#     for model, name in models:
#         print(f"\nTraining {name} model...")
#         save_path = os.path.join(MODELS_DIR, f"{name}_model.pth")
#         train(model, train_loader, test_loader, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=LEARNING_RATE), EPOCHS, save_path)
#         print(f"\nEvaluating {name} model")
#         model.load_state_dict(torch.load(save_path))
#         evaluate(model, test_loader, unique_sites)

# if __name__ == "__main__":
#     main()
