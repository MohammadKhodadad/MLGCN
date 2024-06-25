from utils.model import *
from utils.dataloader import *
from utils.trainer import *


train_loader,test_loader = load_data()
model = Model().to(device)
# Assuming optimizer is defined
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)

# Train the model
train(model, train_loader, test_loader, optimizer)

# Evaluate the model on the test dataset
test_loss, test_acc = test(model, test_loader)