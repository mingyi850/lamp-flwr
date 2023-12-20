import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding, AdamW, get_scheduler
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
import datetime

np.random.seed(100)
torch.manual_seed(100)
device = 'cuda'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cola', 'sst2', 'rotten_tomatoes'], default='cola')
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--noise', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    seq_key = 'text' if args.dataset == 'rotten_tomatoes' else 'sentence'
    num_labels = 2

    # Initialize Distributed Sys
    dist.init_process_group(backend='nccl', init_method='env://')

    device = torch.device("cuda", args.local_rank)

    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model.to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    tokenizer.model_max_length = 512
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.dataset == 'cola':
        train_metric = load_metric('matthews_correlation')
    else:
        train_metric = load_metric('accuracy')

    def tokenize_function(examples):
        return tokenizer(examples[seq_key], truncation=True)


    if args.dataset in ['cola', 'sst2', 'rte']:
        datasets = load_dataset('glue', args.dataset)
    else:
        datasets = load_dataset(args.dataset)

    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    if args.dataset == 'cola' or args.dataset == 'sst2':
        tokenized_datasets = tokenized_datasets.remove_columns(['idx', 'sentence'])
    elif args.dataset == 'rotten_tomatoes':
        tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    else:
        assert False
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')

    tokenized_datasets.set_format('torch')

    train_dataset = tokenized_datasets['train']

    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                              collate_fn=data_collator, num_workers=0)

    opt = AdamW(model.parameters(), lr=5e-5)

    num_training_steps = 100 * len(train_loader)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=opt,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    n_steps = 0
    train_loss = 0

    should_break = False

    for epoch in range(100):
        if should_break:
            break
        train_sampler.set_epoch(epoch)
        model.train()

        for batch in train_loader:
            start_time = datetime.datetime.now()

            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            train_metric.add_batch(predictions=predictions, references=batch['labels'])

            loss = outputs.loss.mean()
            train_loss += loss.item()
            loss.backward()

            if args.noise is not None:
                for param in model.parameters():
                    param.grad.data += torch.randn(param.grad.shape).to(device) * args.noise

            opt.step()
            lr_scheduler.step()
            opt.zero_grad()
            n_steps += 1
            print(n_steps)

            if n_steps % 100 == 0:
                end_time = datetime.datetime.now()
                step_duration = (end_time - start_time).total_seconds()
                print(f"Epoch: {epoch}, Step: {n_steps}, Rank: {args.local_rank}, Step Duration: {step_duration}s")
                print('metric train: ', train_metric.compute())
                print('loss train: ', train_loss / 100)
                train_loss = 0.0

            if n_steps==args.num_steps:
                print('Begin Saving!')

            if n_steps % args.save_every == 0 and n_steps>=args.num_steps:
                if args.local_rank==0:
                    model.module.save_pretrained(f'SYNC/{args.dataset}/noise_{args.noise}/{n_steps}')

            if n_steps == args.num_steps+20:
                print('End')
                should_break = True
                break

if __name__ == '__main__':
    main()


