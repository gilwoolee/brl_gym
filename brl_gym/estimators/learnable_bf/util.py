import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from brl_gym.estimators.learnable_bf.learnable_bf import LearnableBF, BayesFilterNet,  generate_labels
from brl_gym.estimators.learnable_bf.bf_dataset import BayesFilterDataset
from brl_gym.estimators.learnable_bf import pt_util

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    # to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(model, device, optimizer, train_loader, lr, epoch, log_interval, batch_size, mse_loss=False):
    model.train()
    losses = []
    hidden = None # model.init_hidden(batch_size, device)
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        # Separates the hidden state across batches.
        # Otherwise the backward would try to go all the way to the beginning every time.
        if hidden is not None:
            hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden = model(data) # TODO: Pass hidden??
        if mse_loss:
            loss = model.mse_loss(output, label)
        else:
            loss = model.loss(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)


def test(model, device, test_loader, mse_loss=False):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        hidden = None
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output, hidden = model(data, hidden)
            pred = output.max(-1)[1]

            if mse_loss:
                test_loss += model.mse_loss(output, label, reduction='mean').item()
                correct_mask = pred.eq(label.max(-1)[1])
                label = label.max(-1)[1]
            else:
                test_loss += model.loss(output, label, reduction='mean').item()
                correct_mask = pred.eq(label.view_as(pred))

            num_correct = correct_mask.sum().item()
            correct += num_correct
            # Comment this out to avoid printing test results

            if batch_idx % 10 == 0:
                # print('Input\t%s\nGT\t%s\npred\t%s\n\n' % (
                    # np.around(data[0].detach().cpu().numpy(),2),
                print('GT\t%s\npred\t%s\n\n' % (
                    label[0],
                    pred[0]))


    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / (len(test_loader.dataset) * test_loader.dataset.sequence_length)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset) * test_loader.dataset.sequence_length,
    #    100. * correct / (len(test_loader.dataset) * test_loader.dataset.sequence_length)))
    return test_loss, test_accuracy


def main():
    SEQUENCE_LENGTH = 4
    BATCH_SIZE = 10
    FEATURE_SIZE = 15
    TEST_BATCH_SIZE = 5
    EPOCHS = 50
    LEARNING_RATE = 0.002
    WEIGHT_DECAY = 0.0005
    USE_CUDA = True
    PRINT_INTERVAL = 10
    LOG_PATH = 'logs/log.pkl'

    # Generate random data
    output_dim = 10
    input_dim = FEATURE_SIZE

    train_data = np.arange(1800).reshape(20, 6, -1)
    train_label = np.repeat(np.arange(5), 4)
    test_data = np.arange(1800).reshape(20, 6, -1)
    test_label = np.repeat(np.arange(5), 4)

    print('------\n', train_data[0])
    data_train = BayesFilterDataset(train_data, train_label, output_dim, SEQUENCE_LENGTH)
    data_test = BayesFilterDataset(test_data, test_label, output_dim, SEQUENCE_LENGTH)

    print("train has ", len(data_train))
    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    import multiprocessing
    num_workers = multiprocessing.cpu_count()
    print('num workers:', num_workers)

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                               shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE,
                                              shuffle=False, **kwargs)

    model = BayesFilterNet(FEATURE_SIZE, output_dim).to(device)

    # Adam is an optimizer like SGD but a bit fancier. It tends to work faster and better than SGD.
    # We will talk more about different optimization methods in class.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    start_epoch = model.load_last_model('checkpoints')

    train_losses, test_losses, test_accuracies = pt_util.read_log(LOG_PATH, ([], [], []))
    test_loss, test_accuracy = test(model, device, test_loader)

    test_losses.append((start_epoch, test_loss))
    test_accuracies.append((start_epoch, test_accuracy))

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            lr = LEARNING_RATE * np.power(0.25, (int(epoch / 6)))
            train_loss = train(model, device, optimizer, train_loader, lr, epoch, PRINT_INTERVAL)
            test_loss, test_accuracy = test(model, device, test_loader)
            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))
            test_accuracies.append((epoch, test_accuracy))
            pt_util.write_log(LOG_PATH, (train_losses, test_losses, test_accuracies))
            model.save_best_model(test_accuracy, 'checkpoints/%03d.pt' % epoch)
            seed_sequence, seed_label = data_train[0]
            print("seed", seed_sequence, '\n', seed_label[0])
            generated_labels = generate_labels(model, device, seed_sequence, 'max')
            print('generated max\t\t', generated_labels)
            for ii in range(10):
                generated_labels = generate_labels(model, device, seed_sequence, 'sample')
                print('generated sample\t', generated_labels)
            # print('')

    except KeyboardInterrupt as ke:
        print('Interrupted')
    except:
        import traceback
        traceback.print_exc()
    finally:
        print('Saving final model')
        model.save_model('checkpoints/%03d.pt' % epoch, 0)
        ep, val = zip(*train_losses)
        pt_util.plot(ep, val, 'Train loss', 'Epoch', 'Error')
        ep, val = zip(*test_losses)
        pt_util.plot(ep, val, 'Test loss', 'Epoch', 'Error')
        ep, val = zip(*test_accuracies)
        pt_util.plot(ep, val, 'Test accuracy', 'Epoch', 'Error')

        # Plot perplexity
        ep, val = zip(*train_losses)
        val = np.exp(val)
        pt_util.plot(ep, val, 'Train perplexity', 'Epoch', 'Error')
        ep, val = zip(*test_losses)
        val = np.exp(val)
        pt_util.plot(ep, val, 'Test perplexity', 'Epoch', 'Error')
        print("Final test perplexity was ", val[-1])

        return model, device

# final_model, device = main()
