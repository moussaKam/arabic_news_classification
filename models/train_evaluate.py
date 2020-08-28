from torch import optim
import data_loader
import time
from torch import nn
import torch
from model import Model
import argparse
import os

def evaluate(model, data_loader):
    count_batch = 0
    accuracy = 0
    for batch in data_loader:
        sequences = batch[0]
        target = batch[1]
        out, _ = model.forward(sequences)
        predicted = torch.argmax(out, -1)
        accuracy += torch.sum(predicted==target).item()/(sequences.shape[-1])
        count_batch += 1
    accuracy = accuracy/count_batch
    return accuracy

def loadModel(model, optimizer, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_validation_accuracy  = checkpoint['validation_accuracy']
    loss = checkpoint['loss']
    return epoch, best_validation_accuracy, loss


def trainModel(model, path_documents_train, path_labels_train, path_documents_valid,
               path_labels_valid, word2ind, checkpoints_path, lr=0.001, n_epochs=5, 
               batch_size=16, patience=2, printEvery=100, model_path = None):
    epoch = 1
    loss = 0
    count_iter = 0
    epoch_patience = 0 # before interrupting training 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #negative log likelihood
    criterion = nn.NLLLoss()
    best_validation_accuracy = 0 # to decide whether to checkpoint or not
    best_model_checkpoint = 'best_checkpoint.pt'
    # load model from checkpoint
    if model_path != None:
        epoch, best_validation_accuracy, loss = loadModel(model,optimizer, model_path)

    model.init_hidden(batch_size)
    time1 = time.time()
    data_loader_train_params = (path_documents_train, path_labels_train, word2ind, str(model.device), batch_size)
    data_loader_valid_params = (path_documents_valid, path_labels_valid, word2ind, str(model.device), batch_size)
    training_accuracy_epochs = [] # save training accuracy for each epoch
    validation_accuracy_epochs = [] # save validation accuracy for each epoch 
    for i in range(epoch, n_epochs):
        print('-----EPOCH{}-----'.format(i))
        loader = data_loader.get_loader(*data_loader_train_params)
        for batch in loader:
            loss += trainOneBatch(model, batch, optimizer, criterion)
            count_iter += 1
            if count_iter % printEvery == 0:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                          time2 - time1, loss/printEvery))
                loss = 0
        training_accuracy = evaluate(model, data_loader.get_loader(*data_loader_train_params)) 
        validation_accuracy = evaluate(model, data_loader.get_loader(*data_loader_valid_params)) 
        training_accuracy_epochs.append(training_accuracy)
        validation_accuracy_epochs.append(validation_accuracy)
        best_validation_accuracy = max(best_validation_accuracy, validation_accuracy)
        print('Epoch {0} done: training_accuracy = {1:.3f}, validation_accuracy = {2:.3f}'.format(i, training_accuracy, validation_accuracy))
        if validation_accuracy == best_validation_accuracy:
#             best_model_checkpoint = 'checkpoint_epoch_{}.pt'.format(i)
            print('validation accuracy improved: saving checkpoint...')
            if not os.path.isdir(checkpoints_path):
                os.mkdir(checkpoints_path)
            save_path = os.path.join(checkpoints_path, 'best_checkpoint.pt')
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'validation_accuracy': validation_accuracy
            }, save_path)
            print('checkpoint saved to: {}'.format(save_path))
            epoch_patience = 0
        else:
            epoch_patience += 1
            if epoch_patience == patience:
                print('Validation is not improving. stopping training')
                loadModel(model, optimizer, os.path.join(checkpoints_path, 'best_checkpoint.pt'))
                break
    torch.save(model.state_dict(), os.path.join(checkpoints_path, 'model.pt'))
     

def trainOneBatch(model, batch_input, optimizer, criterion):
    optimizer.zero_grad()
    sequences = batch_input[0] # get input sequence of shape: batch_size * sequence_len
    targets = batch_input[1] # get targets of shape : batch_size
    out, A = model.forward(sequences) # shape: batch_size * number_classes
    frobenius = 0
    if model.regularization_coeff > 0:
        aat = torch.bmm(A, A.permute(0,2,1))
        identity = torch.eye(A.shape[1], device=torch.device(model.device)).expand(A.shape[0],-1,-1)
        frobenius = model.regularization_coeff * torch.sum((aat - identity)**2) / A.shape[0]   
    loss = criterion(out, targets) + frobenius
    loss.backward() # compute the gradient
    optimizer.step() # update network parameters
    return loss.item() # return loss value

# def main(args):
#     model = Model(config.embedding_size, config.hidden_size, config.d_a,
#              config.number_attention, config.vocab_size + 2, config.max_seq_len, 5)
#     model = model.to(torch.device(config.device))
#     trainModel(model, model_path=args.checkpoint_path)


# if __name__=='__main__':
#     parse = argparse.ArgumentParser()
#     parse.add_argument('--checkpoint_path', type=str, default=None)
#     args = parse.parse_args()
#     main(args)