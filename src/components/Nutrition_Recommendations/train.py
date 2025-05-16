import torch
from .model import GCN

def train_model(data, input_dim, output_dim, epochs=300, lr=0.001, weight_decay=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(input_dim=input_dim, hidden_dim=64, output_dim=output_dim).to(device)
    data = data.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return model