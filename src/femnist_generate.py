import torch

data_dir = '../data/femnist/'
train_dataset = torch.load(data_dir + 'training.pt')
test_dataset = torch.load(data_dir + 'test.pt')
train_writer_start = 0
test_writer_start = 0
writer_num_list = []
X = torch.tensor([])
Y = torch.tensor([])
for i in range(len(train_dataset[2])):
    train_writer_end = train_writer_start + train_dataset[2][i]
    test_writer_end = test_writer_start + test_dataset[2][i]
    X_train = train_dataset[0][train_writer_start: train_writer_end]
    X_test = test_dataset[0][test_writer_start: test_writer_end]
    Y_train = train_dataset[1][train_writer_start: train_writer_end]
    Y_test = test_dataset[1][test_writer_start: test_writer_end]
    X_writer = torch.cat((X_train, X_test), dim=0)
    Y_writer = torch.cat((Y_train, Y_test), dim=0)
    X = torch.cat((X, X_writer), dim=0)
    Y = torch.cat((Y, Y_writer), dim=0)
    train_writer_start = train_dataset[2][i]
    test_writer_start = test_dataset[2][i]
    writer_num = train_writer_start + test_writer_start
    writer_num_list.append(writer_num)
dataset = tuple([X, Y, writer_num_list])
torch.save(dataset, data_dir + 'femnist.pt')