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
from utils import to_var, idx2word, experiment_name
from new_model import EncoderVAE, DecoderVAE
import math

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
    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

    encoderVAE = EncoderVAE(
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
    
    decoderVAE = DecoderVAE(
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
        encoderVAE = encoderVAE.cuda()
        decoderVAE = decoderVAE.cuda()

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, experiment_name(args,ts)))
        #writer.add_text("model", str(mode))
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

    encoderOptimizer = torch.optim.Adam(encoderVAE.parameters(), lr=args.learning_rate)
    decoderOptimizer = torch.optim.Adam(decoderVAE.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    for epoch in range(args.epochs):

        for split in splits:

            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )
            
            totalIterations = (int(len(datasets[split])/args.batch_size) + 1)*args.epochs

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                encoderVAE.train()
                decoderVAE.train()
            else:
                encoderVAE.eval()
                decoderVAE.eval()

            for iteration, batch in enumerate(data_loader):

                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                hidden, mean, logv, z = encoderVAE(batch['input'], batch['length'])

                # loss calculation
                logp = decoderVAE(batch['input'], batch['length'], hidden)
                
                recon_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                    batch['length'], mean, logv, args.anneal_function, step, totalIterations, split)

                if split == 'train':
                    loss = (recon_loss + KL_weight * KL_loss)/batch_size
                    negELBO = loss
                else:
                    # report complete elbo when validation
                    loss = (recon_loss + KL_loss)/batch_size
                    negELBO = loss

                # backward + optimization
                if split == 'train':
                    encoderOptimizer.zero_grad()
                    decoderOptimizer.zero_grad()
                    loss.backward()
                    if(step < 500):
                        encoderOptimizer.step()
                    else:
                        encoderOptimizer.step()
                        decoderOptimizer.step()
                        
                    #optimizer.step()
                    step += 1


                # bookkeepeing
                tracker['negELBO'] = torch.cat((tracker['negELBO'], negELBO.data))

                if args.tensorboard_logging:
                    writer.add_scalar("%s/Negative_ELBO"%split.upper(), negELBO.data[0], epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/Recon_Loss"%split.upper(), recon_loss.data[0]/batch_size, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL_Loss"%split.upper(), KL_loss.data[0]/batch_size, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL_Weight"%split.upper(), KL_weight, epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    logger.info("%s Batch %04d/%i, Loss %9.4f, Recon-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                        %(split.upper(), iteration, len(data_loader)-1, negELBO.data[0], recon_loss.data[0]/batch_size, KL_loss.data[0]/batch_size, KL_weight))

                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    tracker['target_sents'] += idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx)
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            logger.info("%s Epoch %02d/%i, Mean Negative ELBO %9.4f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['negELBO'])))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/NegELBO"%split.upper(), torch.mean(tracker['negELBO']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents':tracker['target_sents'], 'z':tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/'+ts)
                with open(os.path.join('dumps/'+ts+'/valid_E%i.json'%epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # save checkpoint
            # if split == 'train':
            #     checkpoint_path = os.path.join(save_model_path, "E%i.pytorch"%(epoch))
            #     torch.save(model.state_dict(), checkpoint_path)
            #     logger.info("Model saved at %s"%checkpoint_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    import env as config

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
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
