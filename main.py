import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class PoliksDNN(nn.Module):
    def __init__(self, input_dim, hidden_size, activat, drop_rate, batchnorm):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_size in hidden_size:
            layers.append(nn.Linear(in_dim, h_size))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h_size))
            layers.append(activat)
            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))
            in_dim = h_size
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
from preprocessing import train_clean, test_clean

orig_test = pd.read_csv('data/test.csv')

X_train = train_clean.drop(columns=['Survived', 'PassengerId'], errors='ignore')
y_train = train_clean['Survived']
X_test = test_clean[X_train.columns]

hidden_size = [128, 64, 32, 16]
drop_rate = 0.2
batch_size = 64
epochs = 50
lr = 0.01

X_tensor = torch.tensor(X_train.values, dtype = torch.float32)
y_tensor = torch.tensor(y_train.values, dtype = torch.float32).unsqueeze(1)
    
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
        
model = PoliksDNN(
    input_dim = X_train.shape[1], 
    hidden_size = hidden_size, 
    activat = nn.GELU(), 
    drop_rate = drop_rate, 
    batchnorm = True
    )

loss_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
model.train() 
for epoch in range(epochs):
    for x_, y_ in loader:
        optimizer.zero_grad()
        out = model(x_)
        loss = loss_func(out, y_)
        loss.backward()
        optimizer.step()
    scheduler.step()
            
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test.values, dtype = torch.float32)
    test_probs = torch.sigmoid(model(X_test_tensor)).numpy().flatten()
    predictions = (test_probs > 0.5).astype(int)
    y_true = y_tensor.numpy().flatten()

    submission = pd.DataFrame({
    'PassengerId': orig_test['PassengerId'],
    'Survived': predictions
})

submission['Survived'] = submission['Survived'].astype(int)

submission.to_csv('submission_baseline.csv', index = False)