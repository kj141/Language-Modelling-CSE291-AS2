import os
import json
import time
import torch
import argparse
import logging
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from ptb import PTB
from mixed import Mixed
from utils import to_var, idx2word, experiment_name
from model import SentenceVAE
import math
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

def scatter(x, colors, labelsOfInterest, figSize, plotName):
    # We choose a color palette with seaborn.
    colors = np.array(colors)
    palette = np.array(sns.color_palette("hls", max(labelsOfInterest)+1))

    # We create a scatter plot.
    f = plt.figure(figsize=figSize)
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.title(plotName)
    #ax.legend([plotName])
    ax.axis('off')
    ax.axis('tight')
    
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main(args):

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train', 'valid'] + (['test'] if args.test else [])

    datasets = OrderedDict()
    curBest = 1000000
    for split in splits:
        datasets[split] = Mixed(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

    model = SentenceVAE(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, experiment_name(args,ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    def kl_anneal_function(anneal_function, step, totalIterations, split):
        if(split != 'train'):
            return 1
        elif anneal_function == 'identity':
            return 1
        elif anneal_function == 'linear':
            return 1.005*float(step)/totalIterations
        elif anneal_function == 'sigmoid':
            return (1/(1 + math.exp(-8*(float(step)/totalIterations))))
        elif anneal_function == 'tanh':
            return math.tanh(4*(float(step)/totalIterations))
        elif anneal_function == 'linear_capped':
            #print(float(step)*30/totalIterations)
            return min(1.0, float(step)*5/totalIterations)
        elif anneal_function == 'cyclic':
            quantile = int(totalIterations/5)
            remainder = int(step % quantile)
            midPoint = int(quantile/2)
            if(remainder > midPoint):
              return 1
            else:
              return float(remainder)/midPoint 
        else:
            return 1

    ReconLoss = torch.nn.NLLLoss(size_average=False, ignore_index=datasets['train'].pad_idx)
    def loss_fn(logp, target, length, mean, logv, anneal_function, step, totalIterations, split):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).data[0]].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        
        # Negative Log Likelihood
        recon_loss = ReconLoss(logp, target)

        # KL Divergence
        #print((1 + logv - mean.pow(2) - logv.exp()).size())

        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        #print(KL_loss.size())
        KL_weight = kl_anneal_function(anneal_function, step, totalIterations, split)

        return recon_loss, KL_loss, KL_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    tensor2 = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    tensor3 = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    tensor4 = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    step = 0
    stop = False
    Z = []
    L = []
    for epoch in range(args.epochs):
        if(stop):
            break
        for split in splits:
            if(split == 'test'):
                z_data = []
                domain_label = []
                z_bool = False
                domain_label_bool = False
            if(stop):
                break
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )
            
            totalIterations = (int(len(datasets[split])/args.batch_size) + 1)*args.epochs

            tracker = defaultdict(tensor)
            tracker2 = defaultdict(tensor2)
            tracker3 = defaultdict(tensor3)
            tracker4 = defaultdict(tensor4)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            for iteration, batch in enumerate(data_loader):
#                 if(iteration > 400):
#                     break
                batch_size = batch['input'].size(0)
                labels = batch['label']

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                logp, mean, logv, z = model(batch['input'], batch['length'])
                if(split == 'test'):
                    if(z_bool == False):
                        z_bool = True
                        domain_label = labels.tolist()
                        z_data = z
                    else:
                        domain_label += labels.tolist()
                        #print(domain_label)
                        z_data = torch.cat((z_data, z), 0)

                # loss calculation
                recon_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                    batch['length'], mean, logv, args.anneal_function, step, totalIterations, split)

                if split == 'train':
                    #KL_loss_thresholded = torch.clamp(KL_loss, min=6.0)
                    loss = (recon_loss + KL_weight * KL_loss)/batch_size
                else:
                    # report complete elbo when validation
                    loss = (recon_loss + KL_loss)/batch_size

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1
                
                # bookkeepeing
                tracker['negELBO'] = torch.cat((tracker['negELBO'], loss.data))
                tracker2['KL_loss'] = torch.cat((tracker2['KL_loss'], KL_loss.data))
                tracker3['Recon_loss'] = torch.cat((tracker3['Recon_loss'], recon_loss.data))
                tracker4['Perplexity'] = torch.cat((tracker4['Perplexity'], torch.exp(recon_loss.data/batch_size)))


                if args.tensorboard_logging:
                    writer.add_scalar("%s/Negative_ELBO"%split.upper(), loss.data[0], epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/Recon_Loss"%split.upper(), recon_loss.data[0]/batch_size, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL_Loss"%split.upper(), KL_loss.data[0]/batch_size, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL_Weight"%split.upper(), KL_weight, epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    logger.info("%s Batch %04d/%i, Loss %9.4f, Recon-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                        %(split.upper(), iteration, len(data_loader)-1, loss.data[0], recon_loss.data[0]/batch_size, KL_loss.data[0]/batch_size, KL_weight))
                    
                if(split == 'test'):
                    Z = z_data
                    L = domain_label
                    
                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    tracker['target_sents'] += idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx)
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            logger.info("%s Epoch %02d/%i, Mean Negative ELBO %9.4f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['negELBO'])))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/NegELBO"%split.upper(), torch.mean(tracker['negELBO']), epoch)
                writer.add_scalar("%s-Epoch/KL_loss"%split.upper(), torch.mean(tracker2['KL_loss'])/batch_size, epoch)
                writer.add_scalar("%s-Epoch/Recon_loss"%split.upper(), torch.mean(tracker3['Recon_loss'])/batch_size, epoch)
                writer.add_scalar("%s-Epoch/Perplexity"%split.upper(), torch.mean(tracker4['Perplexity']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                if(torch.mean(tracker['negELBO']) < curBest):
                    curBest = torch.mean(tracker['negELBO'])
                else:
                    stop = True
                dump = {'target_sents':tracker['target_sents'], 'z':tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps_32_0', ts)):
                    os.makedirs('dumps_32_0/'+ts)
                with open(os.path.join('dumps_32_0/'+ts+'/valid_E%i.json'%epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # save checkpoint
            # if split == 'train':
            #     checkpoint_path = os.path.join(save_model_path, "E%i.pytorch"%(epoch))
            #     torch.save(model.state_dict(), checkpoint_path)
            #     logger.info("Model saved at %s"%checkpoint_path)
    
    Z = Z.data.cpu().numpy()
    print(Z.shape)
    beforeTSNE = TSNE(random_state=20150101).fit_transform(Z)
    scatter(beforeTSNE, L, [0,1,2], (5,5), 'latent discoveries')
    plt.savefig('mixed_tsne'+args.anneal_function+'.png', dpi=120)
    
    
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    import env as config

    parser.add_argument('--data_dir', type=str, default='data_mixed')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_false')

    parser.add_argument('-ep', '--epochs', type=int, default=40)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, default='linear')

    parser.add_argument('-v','--print_every', type=int, default=50)
    parser.add_argument('-tb','--tensorboard_logging', action='store_false')
    parser.add_argument('-log','--logdir', type=str, default=config.logs)
    parser.add_argument('-bin','--save_model_path', type=str, default=config.bin_)

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
