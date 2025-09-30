"""
Wrapper to train original models from eeg-gnn-ssl repository 
with patched file handling
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import json
import argparse
from collections import OrderedDict
import copy

# Add the current directory to path for importing other modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import repo modules
try:
    import utils
    from args import get_args
    from constants import INCLUDED_CHANNELS, FREQUENCY
    from model.model import DCRNNModel_classification, DCRNNModel_nextTimePred
    from model.densecnn import DenseCNN
    from model.lstm import LSTMModel
    from model.cnnlstm import CNN_LSTM
    from tensorboardX import SummaryWriter
    from dotted_dict import DottedDict
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the eeg-gnn-ssl repository root.")
    sys.exit(1)

# Patch the original dataloader to fix file path issues
class PatchedDataLoader:
    """Patched data loader that avoids file path issues"""
    def __init__(self, preproc_dir, task='detection', batch_size=4, shuffle=True):
        self.preproc_dir = preproc_dir
        self.task = task
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Find all preprocessed files
        self.file_list = [f for f in os.listdir(preproc_dir) if f.endswith('.npz')]
        print(f"Found {len(self.file_list)} preprocessed files")
        
        # Split into train/val/test sets
        np.random.seed(42)
        indices = np.random.permutation(len(self.file_list))
        train_end = int(len(indices) * 0.7)
        val_end = int(len(indices) * 0.85)
        
        self.train_files = [self.file_list[i] for i in indices[:train_end]]
        self.val_files = [self.file_list[i] for i in indices[train_end:val_end]]
        self.test_files = [self.file_list[i] for i in indices[val_end:]]
        
        print(f"Split into {len(self.train_files)} train, {len(self.val_files)} val, {len(self.test_files)} test")
        
        # Load one file to get data shape
        if len(self.file_list) > 0:
            try:
                sample_data = np.load(os.path.join(preproc_dir, self.file_list[0]))
                self.data_shape = sample_data['data'].shape
                print(f"Data shape: {self.data_shape}")
            except Exception as e:
                print(f"Error loading sample file: {e}")
                self.data_shape = None
    
    def get_loaders(self):
        """Return train, val, test loaders"""
        train_loader = self._create_loader(self.train_files)
        val_loader = self._create_loader(self.val_files, shuffle=False)
        test_loader = self._create_loader(self.test_files, shuffle=False)
        
        return {
            'train': train_loader,
            'dev': val_loader,
            'test': test_loader
        }, self.data_shape
    
    def _create_loader(self, file_list, shuffle=None):
        """Create a DataLoader for a list of files"""
        dataset = PatchedDataset(self.preproc_dir, file_list)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle is None else shuffle,
            num_workers=0,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function to handle missing files and create supports"""
        # Filter out None samples
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        
        # Separate data and labels
        data, labels = zip(*batch)
        
        # Stack data and labels
        data = torch.stack(data)
        labels = torch.stack(labels)
        
        # Create dummy sequence lengths (all same length)
        seq_lengths = torch.ones(data.shape[0], dtype=torch.long) * data.shape[2]
        
        # Create dummy supports (identity matrices)
        num_nodes = data.shape[1]
        supports = [torch.eye(num_nodes)]
        
        # Create dummy file names
        file_names = ["dummy.edf"] * data.shape[0]
        
        return data, labels, seq_lengths, supports, None, file_names

class PatchedDataset(torch.utils.data.Dataset):
    """Dataset for patched dataloader"""
    def __init__(self, preproc_dir, file_list):
        self.preproc_dir = preproc_dir
        self.file_list = file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.preproc_dir, self.file_list[idx])
        
        try:
            data = np.load(file_path)
            x = data['data']
            y = int(data['label'])
            
            # Convert to torch tensors
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            
            return x, y
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

def train(model, dataloaders, args, device, save_dir, log, tbx):
    """
    Perform training and evaluate on val set
    Based on original train function but with modifications
    """
    # Define loss function
    if args.task == 'detection':
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    # Data loaders
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']

    # Get saver
    saver = utils.CheckpointSaver(save_dir,
                               metric_name=args.metric_name,
                               maximize_metric=args.maximize_metric,
                               log=log)

    # To train mode
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.Adam(params=model.parameters(),
                        lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # average meter for validation loss
    nll_meter = utils.AverageMeter()

    # Train
    log.info('Training...')
    epoch = 0
    step = 0
    prev_val_loss = 1e10
    patience_count = 0
    early_stop = False
    while (epoch != args.num_epochs) and (not early_stop):
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        total_samples = len(train_loader.dataset)
        with torch.enable_grad(), \
                tqdm(total=total_samples) as progress_bar:
            for batch in train_loader:
                # Skip empty batches
                if batch is None:
                    continue
                
                x, y, seq_lengths, supports, _, _ = batch
                batch_size = x.shape[0]

                # input seqs
                x = x.to(device)
                y = y.to(device)  # (batch_size,)
                seq_lengths = seq_lengths.to(device)  # (batch_size,)
                for i in range(len(supports)):
                    supports[i] = supports[i].to(device)

                # Zero out optimizer first
                optimizer.zero_grad()

                # Forward
                # (batch_size, num_classes)
                if args.model_name == "dcrnn":
                    logits = model(x, seq_lengths, supports)
                elif args.model_name == "densecnn":
                    x = x.transpose(-1, -2).reshape(batch_size, -1, args.num_nodes) # (batch_size, seq_len, num_nodes)
                    logits = model(x)
                elif args.model_name == "lstm" or args.model_name == "cnnlstm":
                    logits = model(x, seq_lengths)
                else:
                    raise NotImplementedError
                
                if logits.shape[-1] == 1:
                    logits = logits.view(-1)  # (batch_size,)
                    
                loss = loss_fn(logits, y)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                step += batch_size

                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss_val,
                                         lr=optimizer.param_groups[0]['lr'])

                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

            if epoch % args.eval_every == 0:
                # Evaluate and save checkpoint
                log.info('Evaluating at epoch {}...'.format(epoch))
                eval_results = evaluate(model,
                                        dev_loader,
                                        args,
                                        save_dir,
                                        device,
                                        is_test=False,
                                        nll_meter=nll_meter)
                best_path = saver.save(epoch,
                                       model,
                                       optimizer,
                                       eval_results[args.metric_name])

                # Accumulate patience for early stopping
                if eval_results['loss'] < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results['loss']

                # Early stop
                if patience_count == args.patience:
                    early_stop = True

                # Back to train mode
                model.train()

                # Log to console
                results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                        for k, v in eval_results.items())
                log.info('Dev {}'.format(results_str))

                # Log to TensorBoard
                log.info('Visualizing in TensorBoard...')
                for k, v in eval_results.items():
                    tbx.add_scalar('eval/{}'.format(k), v, step)

        # Step lr scheduler
        scheduler.step()

def evaluate(model, dataloader, args, save_dir, device, is_test=False, nll_meter=None, eval_set='dev', best_thresh=0.5):
    """
    Modified evaluate function to work with patched dataloader
    """
    # To evaluate mode
    model.eval()

    # Define loss function
    if args.task == 'detection':
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    y_pred_all = []
    y_true_all = []
    y_prob_all = []
    file_name_all = []
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for batch in dataloader:
            # Skip empty batches
            if batch is None:
                continue
                
            x, y, seq_lengths, supports, _, file_name = batch
            batch_size = x.shape[0]

            # Input seqs
            x = x.to(device)
            y = y.to(device)  # (batch_size,)
            seq_lengths = seq_lengths.to(device)  # (batch_size,)
            for i in range(len(supports)):
                supports[i] = supports[i].to(device)

            # Forward
            # (batch_size, num_classes)
            if args.model_name == "dcrnn":
                logits = model(x, seq_lengths, supports)
            elif args.model_name == "densecnn":
                x = x.transpose(-1, -2).reshape(batch_size, -1, args.num_nodes) # (batch_size, len*freq, num_nodes)
                logits = model(x)
            elif args.model_name == "lstm" or args.model_name == "cnnlstm":
                logits = model(x, seq_lengths)
            else:
                raise NotImplementedError

            if args.num_classes == 1:  # binary detection
                logits = logits.view(-1)  # (batch_size,)
                y_prob = torch.sigmoid(logits).cpu().numpy()  # (batch_size, )
                y_true = y.cpu().numpy().astype(int)
                y_pred = (y_prob > best_thresh).astype(int)  # (batch_size, )
            else:
                # (batch_size, num_classes)
                y_prob = F.softmax(logits, dim=1).cpu().numpy()
                y_pred = np.argmax(y_prob, axis=1).reshape(-1)  # (batch_size,)
                y_true = y.cpu().numpy().astype(int)

            # Update loss
            loss = loss_fn(logits, y)
            if nll_meter is not None:
                nll_meter.update(loss.item(), batch_size)

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
            file_name_all.extend(file_name)

            # Log info
            progress_bar.update(batch_size)

    # Concatenate results
    if y_pred_all:
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)
    else:
        # Handle empty case
        y_pred_all = np.array([])
        y_true_all = np.array([])
        y_prob_all = np.array([])

    # Threshold search, for detection only
    if (args.task == "detection") and (eval_set == 'dev') and is_test and len(y_true_all) > 0:
        best_thresh = utils.thresh_max_f1(y_true=y_true_all, y_prob=y_prob_all)
        # update dev set y_pred based on best_thresh
        y_pred_all = (y_prob_all > best_thresh).astype(int)  # (batch_size, )
    else:
        best_thresh = best_thresh

    # Calculate scores
    if len(y_true_all) > 0:
        scores_dict, _, _ = utils.eval_dict(y_pred=y_pred_all,
                                        y=y_true_all,
                                        y_prob=y_prob_all,
                                        file_names=file_name_all,
                                        average="binary" if args.task == "detection" else "weighted")
    else:
        # Create dummy scores if no data
        scores_dict = {
            'acc': 0,
            'F1': 0,
            'recall': 0,
            'precision': 0
        }
        if args.task == "detection":
            scores_dict['auroc'] = 0.5
    
    # Get evaluation loss
    eval_loss = nll_meter.avg if (nll_meter is not None) else loss.item() if 'loss' in locals() else 0.0
    
    # Collect and return results
    results_list = [('loss', eval_loss),
                  ('acc', scores_dict['acc']),
                  ('F1', scores_dict['F1']),
                  ('recall', scores_dict['recall']),
                  ('precision', scores_dict['precision']),
                  ('best_thresh', best_thresh)]
    
    if 'auroc' in scores_dict.keys():
        results_list.append(('auroc', scores_dict['auroc']))
    
    results = OrderedDict(results_list)

    return results

def main():
    # Parse arguments using original args parser
    args = get_args()
    
    # Force some arguments to work with our patched version
    args.max_seq_len = 12  # Use 12-second clips
    args.use_fft = True    # Use FFT features
    args.data_augment = True
    
    # Set device
    args.cuda = torch.cuda.is_available()
    device = "cuda" if args.cuda else "cpu"
    print(f"Using device: {device}")
    
    # Set random seed
    utils.seed_torch(seed=args.rand_seed)
    
    # Set up save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, training=True if args.do_train else False)
    
    # Save args
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
    
    # Set up logger
    log = utils.get_logger(args.save_dir, 'train')
    tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(json.dumps(vars(args), indent=4, sort_keys=True)))
    
    # Build patched dataset
    log.info('Building patched dataset...')
    data_loader = PatchedDataLoader(
        preproc_dir=args.preproc_dir,
        task=args.task,
        batch_size=args.train_batch_size
    )
    dataloaders, data_shape = data_loader.get_loaders()
    
    # Set num_nodes based on data shape
    args.num_nodes = data_shape[1]  # Channels dimension
    
    # Build model
    log.info('Building model...')
    if args.model_name == "dcrnn":
        model = DCRNNModel_classification(
            args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "densecnn":
        with open("./model/dense_inception/params.json", "r") as f:
            params = json.load(f)
        params = DottedDict(params)
        data_shape = (args.max_seq_len*100, args.num_nodes) if args.use_fft else (args.max_seq_len*200, args.num_nodes)
        model = DenseCNN(params, data_shape=data_shape, num_classes=args.num_classes)
    elif args.model_name == "lstm":
        model = LSTMModel(args, args.num_classes, device)
    elif args.model_name == "cnnlstm":
        model = CNN_LSTM(args.num_classes)
    else:
        raise NotImplementedError
    
    # Load pretrained model if specified
    if args.do_train:
        if not args.fine_tune:
            if args.load_model_path is not None:
                model = utils.load_model_checkpoint(
                    args.load_model_path, model)
        else:  # fine-tune from pretrained model
            if args.load_model_path is not None:
                args_pretrained = copy.deepcopy(args)
                setattr(
                    args_pretrained,
                    'num_rnn_layers',
                    args.pretrained_num_rnn_layers)
                pretrained_model = DCRNNModel_nextTimePred(
                    args=args_pretrained, device=device)  # placeholder
                pretrained_model = utils.load_model_checkpoint(
                    args.load_model_path, pretrained_model)

                model = utils.build_finetune_model(
                    model_new=model,
                    model_pretrained=pretrained_model,
                    num_rnn_layers=args.num_rnn_layers)
            else:
                raise ValueError(
                    'For fine-tuning, provide pretrained model in load_model_path!')
    
    # Count parameters
    num_params = utils.count_parameters(model)
    log.info('Total number of trainable parameters: {}'.format(num_params))
    
    # Move model to device
    model = model.to(device)
    
    # Train
    if args.do_train:
        train(model, dataloaders, args, device, args.save_dir, log, tbx)
        
        # Load best model after training
        best_path = os.path.join(args.save_dir, 'best.pth.tar')
        model = utils.load_model_checkpoint(best_path, model)
        model = model.to(device)
    
    # Evaluate
    log.info('Training DONE. Evaluating model...')
    dev_results = evaluate(model,
                         dataloaders['dev'],
                         args,
                         args.save_dir,
                         device,
                         is_test=True,
                         nll_meter=None,
                         eval_set='dev')

    dev_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                              for k, v in dev_results.items())
    log.info('DEV set prediction results: {}'.format(dev_results_str))

    test_results = evaluate(model,
                          dataloaders['test'],
                          args,
                          args.save_dir,
                          device,
                          is_test=True,
                          nll_meter=None,
                          eval_set='test',
                          best_thresh=dev_results['best_thresh'])

    # Log to console
    test_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                               for k, v in test_results.items())
    log.info('TEST set prediction results: {}'.format(test_results_str))

if __name__ == "__main__":
    main()