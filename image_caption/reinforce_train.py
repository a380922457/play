import torch
import torch.optim as optim
from torch.autograd import Variable

# import eval_utils
import misc.utils as utils
from data_loader import get_loader
from misc.rewards import get_self_critical_reward
from models import att2in

model_path = './models/'
crop_size = 224
log_step = 100
save_step = 10000

embed_size = 256
hidden_size = 128
num_layers = 1

weight_decay = 0.9
num_epochs = 5
batch_size = 3
num_workers = 2
learning_rate = 0.001
vocab_size = 12000
use_cuda = False


def train():
    data_loader = get_loader(batch_size, shuffle=False, num_workers=num_workers)
    model = att2in.Att2inModel()
    if use_cuda:
        model.cuda()
    model.train()

    # crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            images = Variable(images, requires_grad=False)
            captions = Variable(captions, requires_grad=False)

            if use_cuda:
                images = images.cuda()
                captions = captions.cuda()

            optimizer.zero_grad()

            # torch.cuda.synchronize()

            gen_result, sample_log_prob = model.sample(images, sample_max=0)
            reward = get_self_critical_reward(model, images, captions, gen_result)
            loss = rl_crit(sample_log_prob, gen_result, Variable(torch.from_numpy(reward).float().cuda(), requires_grad=False))

            loss.backward()
            optimizer.step()
            train_loss = loss.data[0]
            # torch.cuda.synchronize()

train()
