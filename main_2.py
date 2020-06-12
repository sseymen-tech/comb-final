import torch
import torch.nn as nn
import numpy as np

num_epochs =50
learning_rate = 0.05

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out

location = 'D:/Sinan-data/Macular/AGGCN plots/'
# x_train = np.loadtxt('size39_numpy_g_.txt')
# y_train = np.loadtxt('size39_numpy_g_match.txt')
# x_test =  np.loadtxt('bcspwr01_numpy_g_.txt')
# y_test = np.loadtxt('bcspwr01_numpy_g_match.txt')
#
# x_train = np.loadtxt(str(location)+'800_numpy_g_1.txt')
# y_train = np.loadtxt(str(location)+'800_numpy_g_match1.txt')
# x_test =  np.loadtxt(str(location)+'800G15_numpy_g_original.txt')
# y_test = np.loadtxt(str(location)+'800G15_numpy_g_match_original.txt')

# x_train = np.loadtxt(str(location)+'800_numpy_g_1.txt')  ###G17 data
# y_train = np.loadtxt(str(location)+'800_numpy_g_match1.txt')
# x_test =  np.loadtxt(str(location)+'800_numpy_g_original.txt')
# y_test = np.loadtxt(str(location)+'800_numpy_g_match_original.txt')

# x_train = np.loadtxt(str(location)+'72G15_numpy_g0.txt')  ### dwt_72 data
# y_train = np.loadtxt(str(location)+'72G15_numpy_g_match0.txt')
# x_test =  np.loadtxt(str(location)+'72G15_numpy_g_original.txt')
# y_test = np.loadtxt(str(location)+'72G15_numpy_g_match_original.txt')

# x_train = np.loadtxt(str(location)+'198G15_numpy_g0.txt')  ### dwt_198 data
# y_train = np.loadtxt(str(location)+'198G15_numpy_g_match0.txt')
# x_test =  np.loadtxt(str(location)+'198G15_numpy_g_original.txt')
# y_test = np.loadtxt(str(location)+'198G15_numpy_g_match_original.txt')

# x_train = np.loadtxt(str(location)+'2680G15_numpy_g0.txt')  ### dwt_198 data
# y_train = np.loadtxt(str(location)+'2680G15_numpy_g_match0.txt')
# x_test =  np.loadtxt(str(location)+'2680G15_numpy_g_original.txt')
# y_test = np.loadtxt(str(location)+'2680G15_numpy_g_match_original.txt')


# x_train = np.loadtxt(str(location)+'258G15_numpy_g0.txt')  ### sphere data
# y_train = np.loadtxt(str(location)+'258G15_numpy_g_match0.txt')
# x_test =  np.loadtxt(str(location)+'258G15_numpy_g_original.txt')
# y_test = np.loadtxt(str(location)+'258G15_numpy_g_match_original.txt')

# x_train = np.loadtxt(str(location)+'62G15_numpy_g0.txt')  ### can data
# y_train = np.loadtxt(str(location)+'62G15_numpy_g_match0.txt')
# x_test =  np.loadtxt(str(location)+'62G15_numpy_g_original.txt')
# y_test = np.loadtxt(str(location)+'62G15_numpy_g_match_original.txt')

# x_train = np.loadtxt(str(location)+'153G15_numpy_g0.txt')  ### bc05 data
# y_train = np.loadtxt(str(location)+'153G15_numpy_g_match0.txt')
# x_test =  np.loadtxt(str(location)+'153G15_numpy_g_original.txt')
# y_test = np.loadtxt(str(location)+'153G15_numpy_g_match_original.txt')

# x_train = np.loadtxt(str(location)+'39_numpy_g_0.txt')  ### bc01  data
# y_train = np.loadtxt(str(location)+'39_numpy_g_match0.txt')
# x_test =  np.loadtxt(str(location)+'39G15_numpy_g_original.txt')
# y_test = np.loadtxt(str(location)+'39G15_numpy_g_match_original.txt')

# x_train = np.loadtxt(str(location)+'1089_numpy_g_0.txt')  ### b2_ss  data
# y_train = np.loadtxt(str(location)+'1089_numpy_g_match0.txt')
# x_test =  np.loadtxt(str(location)+'1089G15_numpy_g_original.txt')
# y_test = np.loadtxt(str(location)+'1089G15_numpy_g_match_original.txt')

# x_train = np.loadtxt(str(location)+'1440G15_numpy_g0.txt')  ### 1440  data
# y_train = np.loadtxt(str(location)+'1440G15_numpy_g_match0.txt')
# x_test =  np.loadtxt(str(location)+'1440G15_numpy_g_original.txt')
# y_test = np.loadtxt(str(location)+'1440G15_numpy_g_match_original.txt')

# x_train = np.loadtxt(str(location)+'406G15_numpy_g0.txt')  ### 226  data
# y_train = np.loadtxt(str(location)+'406G15_numpy_g_match0.txt')
# x_test =  np.loadtxt(str(location)+'406G15_numpy_g_original.txt')
# y_test = np.loadtxt(str(location)+'406G15_numpy_g_match_original.txt')

x_train = np.loadtxt(str(location)+'472G15_numpy_g0.txt')  ### 226  data
y_train = np.loadtxt(str(location)+'472G15_numpy_g_match0.txt')
x_test =  np.loadtxt(str(location)+'472G15_numpy_g_original.txt')
y_test = np.loadtxt(str(location)+'472G15_numpy_g_match_original.txt')

# x_train = np.loadtxt('train800_numpy_g_.txt')
# y_train = np.loadtxt('train800_numpy_g_match.txt')
# x_test =  np.loadtxt('G17_numpy_g_.txt')
# y_test = np.loadtxt('G17_numpy_g_match.txt')
#
# x_train = np.loadtxt(str(location)+'662_numpy_g_9.txt')
# y_train = np.loadtxt(str(location)+'662_numpy_g_match9.txt')
# x_test =  np.loadtxt(str(location)+'662_numpy_g_original.txt')
# y_test =  np.loadtxt(str(location)+'662_numpy_g_match_original.txt')
#
# x_train = np.loadtxt('5300_numpy_g_.txt')
# y_train = np.loadtxt('5300_numpy_g_match.txt')
# x_test =  np.loadtxt('bcspwr10_numpy_g_.txt')
# y_test = np.loadtxt('bcspwr10_numpy_g_match.txt')

# Linear regression model
inputs = torch.from_numpy(x_train)
targets = torch.from_numpy(y_train)
print(inputs.size())
print(list(inputs.size())[0])
print(inputs.flatten().size())
# x = inputs.flatten().size().numpy()[0]
# print(x)
nnum = list(inputs.size())[0]
print('inputssize',inputs.flatten().size())
print('targetssize',targets.flatten().size())

model = NeuralNet(list(inputs.flatten().size())[0], 200, list(targets.flatten().size())[0])

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
    print(epoch)

    # Forward pass
    outputs = model(inputs.flatten().float())
    loss = criterion(outputs.float(), targets.flatten().float())

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test
model.eval()
with torch.no_grad():
    images = torch.from_numpy(x_test).flatten().float()
    labels = torch.from_numpy(y_test).flatten().float()
    outputs = model(images)
    argsorted = torch.argsort(outputs).numpy()
    M = 0
    selected_nodes = []
    for i in argsorted:
        arg0 = int(i/nnum)
        arg1 = i%nnum
        if arg0 not in selected_nodes and arg1 not in selected_nodes:
            if inputs[arg0][arg1] == 1:
                M += 1
                selected_nodes.append(arg0)
                selected_nodes.append(arg1)
    print(M,torch.sum(labels)/2)
    print(M/(torch.sum(labels)/2))
