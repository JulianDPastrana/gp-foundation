import torch
from torch.utils.data import DataLoader
from chainedgp.datasets.lorenz_attractor import LorenzAttractorDataset
from chainedgp.rnn import MultiRateLSTM
from chainedgp.utils.training import train_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1103
torch.set_default_dtype(torch.float64)


print(f"Using device: {DEVICE}")

# Create Datasets
dataset = LorenzAttractorDataset(num_steps=1000, downsampling_rates=(1, 10, 100))
print(f"Dataset length: {len(dataset)}")
train_ds, valid_ds, test_ds = None, None, None
# print(f"Train: {len(train_ds)}, Validation: {len(valid_ds)}, Test: {len(test_ds)}")
batch_size = 32
# train_loader = make_balanced_loader(train_ds, batch_size)
# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Set Up the model
hidden_size = 15
input_size_list = [1, 2]
model = torch.nn.Sequential(
    MultiRateLSTM(input_size_list=input_size_list, hidden_size=hidden_size),
    torch.nn.Linear(in_features=hidden_size, out_features=1),
)

# Loss function
loss_fn = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training routine
# if True:
#     model_path = train_model(
#         model=model,
#         train_loader=train_loader,
#         validation_loader=valid_loader,
#         EPOCHS=250,
#         model_name="multi_rate_lstm_toadstool",
#         loss_fn=loss_fn,
#         optimizer=optimizer,
#     )
# else:
#     model_path = "models/multi_rate_lstm_toadstool_20250624_111358"
#     print(f"Model saved to {model_path}")
#     model = MultiCellLSTM(hidden_size=hidden_size, num_classes=num_classes).to(DEVICE)
#     model.load_state_dict(torch.load(model_path))
#
# --- Evaluation on test set ---
model.eval()
y_true = []
y_pred = []

# with torch.no_grad():
#     for x, y in test_loader:
#         # Forward pass
#         out = model(x)
#         preds = out.argmax(dim=-1)
#         # Accumulate
#         y_true.extend(y.cpu().tolist())
#         y_pred.extend(preds.cpu().tolist())
