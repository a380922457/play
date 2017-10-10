import torch
import numpy as np
import os
from data_loader import get_loader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import datetime
from attention_model import CaptionModel
from utils import LanguageModelCriterion
from ada_attention_model import AdaAttModel

model_path = './models/'
crop_size = 224
log_step = 100
save_step = 10000

embed_size = 256
hidden_size = 128
num_layers = 1

num_epochs = 5
batch_size = 3
num_workers = 2
learning_rate = 0.001
vocab_size = 12000
use_cuda = False


def main():
    data_loader = get_loader(batch_size, shuffle=False, num_workers=num_workers)
    model = AdaAttModel()

    if use_cuda:
        model.cuda()

    # Loss and Optimizer
    model.train()
    criterion = LanguageModelCriterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Models
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            images = Variable(images, requires_grad=False)
            captions = Variable(captions, requires_grad=False)
            if use_cuda:
                images = images.cuda()
                captions = captions.cuda()
            # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # Forward, Backward and Optimize
            optimizer.zero_grad()
            outputs = model(images, captions)

            mask = torch.from_numpy(np.ones([128, 7]))

            loss = criterion(outputs, captions, mask)
            loss.backward()
            optimizer.step()

            # Print log info
            if i % log_step == 0:
                endtime = datetime.datetime.now()
                if i != 0:
                    print("time_cost", (endtime - starttime).seconds)
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      % (epoch, num_epochs, i, total_step, loss.data[0], np.exp(loss.data[0])))
                starttime = datetime.datetime.now()

                # Save the models
            # if (i + 1) % save_step == 0:
            #     torch.save(decoder.state_dict(), os.path.join(model_path, 'decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
            #     torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder-%d-%d.pkl' % (epoch + 1, i + 1)))


if __name__ == '__main__':
    main()
