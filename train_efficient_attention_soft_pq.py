import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import argparse
import numpy as np
from collections import Counter
import os
import loss
from models.efficient_attention_soft_pq import EfficientAttnSoftPQSASRecModel
from neg_sampler import UniformNegativeSampler
from dataloader_utils import collate_fn, collate_fn_with_negatives, LadderSampler
from utils import generate_padding_mask, unserialize, reset_random_seed


def evaluate(model, eval_config):
    # negative samplers must be providedd
    eval_dataset = eval_config['eval_dataset']
    batch_size = eval_config['batch_size']
    device = eval_config['device']
    num_negatives = eval_config['num_negatives']

    data_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_with_negatives,
        pin_memory=False
    )

    data_iterator = tqdm(enumerate(data_loader),
                         total=len(data_loader), leave=True)
    order_counter = Counter()
    count_array = np.zeros(num_negatives + 1)
    model.eval()
    model.manual_ip_table_update()
    with torch.no_grad():
        for _, (batch_items, batch_targets, batch_seq_lengths) in data_iterator:
            batch_items = batch_items.to(device)
            batch_targets = batch_targets.to(device)
            batch_seq_lengths = batch_seq_lengths.to(device)
            padding_mask = generate_padding_mask(batch_seq_lengths, device)
            # actually no casual mask is required in evaluation if we only use one self-attention layer
            # (N, 1, 1 + num_neg)
            scores = model(batch_items, batch_targets, padding_mask, batch_seq_lengths)
            # (N, 1 + num_neg)
            scores = scores.squeeze(1)
            # (N, 1 + num_neg)
            idx = torch.argsort(scores, dim=1, descending=True)
            # index of item 0 after sorting the (num_neg + 1) items
            pos_item_orders = torch.argmin(idx, dim=1)
            order_counter.update(pos_item_orders.tolist())
    for k, v in order_counter.items():
        count_array[k] = v

    hr = count_array.cumsum()
    ndcg = 1 / np.log2(np.arange(0, num_negatives + 1) + 2)
    ndcg = ndcg * count_array
    ndcg = ndcg.cumsum() / hr.max()
    hr = hr / hr.max()
    return hr[:10], ndcg[:10]


def train(model, training_config):
    train_dataset = training_config['train_dataset']
    negative_sampler = training_config['negative_sampler']
    optimizer = training_config['optimizer']
    loss_fn = training_config['loss_fn']
    device = training_config['device']
    batch_size = training_config['batch_size']
    num_negatives = training_config['num_negatives']
    num_epochs = training_config['num_epochs']
    num_workers = training_config['num_workers']
    num_init_batches_to_update_ip_table = training_config['num_init_batches_to_update_ip_table']
    update_ip_table_per_batch = num_init_batches_to_update_ip_table == -1

    data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch, negative_sampler, k=num_negatives),
        sampler=LadderSampler(train_dataset, batch_size),
        num_workers=num_workers,
        pin_memory=False
    )

    model.train()
    for epoch in range(num_epochs):
        print("=====epoch {:>2d}=====".format(epoch + 1))
        batch_iterator = tqdm(enumerate(data_loader),
                              total=len(data_loader), leave=True)
        num_batches = len(data_loader)
        running_loss = 0.
        start_time = time.time()

        for batch_idx, (batch_items, batch_targets, batch_seq_lengths) in batch_iterator:
            batch_items = batch_items.to(device)
            batch_targets = batch_targets.to(device)
            batch_seq_lengths = batch_seq_lengths.to(device)
            optimizer.zero_grad()
            # (N, L)
            padding_mask = generate_padding_mask(batch_seq_lengths, device)
            # (L, L)
            # (N, L, 1 + num_neg)
            if epoch == 0 and batch_idx < num_init_batches_to_update_ip_table:
                scores, reg_loss = model(batch_items, batch_targets, padding_mask, batch_seq_lengths,
                                         update_ip_table=True)
                # (N, L)
                pos_scores = scores[:, :, 0]
                # (N, L, num_neg)
                neg_scores = scores[:, :, 1:]
                loss = loss_fn(pos_scores, neg_scores)
                loss = torch.sum(loss * padding_mask) / \
                       torch.sum(batch_seq_lengths)
                loss += reg_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if batch_idx == num_init_batches_to_update_ip_table - 1:
                    model.inner_product_table.detach_()
                    if hasattr(model, "positional_embedding_module"):
                        model.inner_product_table_pos_emb.detach_()
            else:
                if not update_ip_table_per_batch and batch_idx == 0:
                    model.manual_ip_table_update()
                    if hasattr(model, "positional_embedding_module"):
                        model.manual_ip_table_update_pos_emb()
                scores, reg_loss = model(batch_items, batch_targets, padding_mask, batch_seq_lengths,
                                         update_ip_table=update_ip_table_per_batch)
                # (N, L)
                pos_scores = scores[:, :, 0]
                # (N, L, num_neg)
                neg_scores = scores[:, :, 1:]
                loss = loss_fn(pos_scores, neg_scores)
                loss = torch.sum(loss * padding_mask) / \
                       torch.sum(batch_seq_lengths)
                loss += reg_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if not update_ip_table_per_batch and batch_idx == 0:
                    model.inner_product_table.detach_()
                    if hasattr(model, "positional_embedding_module"):
                        model.inner_product_table_pos_emb.detach_()
            batch_iterator.set_postfix_str(f"loss={loss.item():.4f}")

        epoch_time = time.time() - start_time
        print("epoch {:>2d} completed.".format(epoch + 1))
        print("time taken: {:.2f} sec".format(epoch_time))
        print("avg. loss: {:.4f}".format(running_loss / num_batches))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, default='')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_save_path", type=str, default='')
    parser.add_argument("--model_load_path", type=str, default='')
    eval_only_mode_parser = parser.add_mutually_exclusive_group(required=False)
    eval_only_mode_parser.add_argument("--train_and_eval", dest='eval_only_mode', action='store_false')
    eval_only_mode_parser.add_argument("--eval_only", dest='eval_only_mode', action='store_true')
    parser.set_defaults(eval_only_mode=False)
    args = parser.parse_args()

    reset_random_seed(42)

    config = unserialize(args.config)
    train_dataset = unserialize(args.train_dataset)
    eval_dataset = unserialize(args.eval_dataset)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loss_fn = loss.__getattribute__(config['training_config']['loss_function'])(
        **config['training_config']['loss_config'])
    sampler = UniformNegativeSampler(train_dataset.num_items, exclude_pos=config["training_config"].get(
        "exclude_positive_in_negative_sampling", False))

    emb_dim = config["model_config"].pop("embedding_dim")
    num_codebooks = config["model_config"]["product_quantization_config"].pop("num_codebooks")
    num_codewords = config["model_config"]["product_quantization_config"].pop("num_codewords")
    dropout = config["model_config"].pop("dropout")
    model = EfficientAttnSoftPQSASRecModel(
        n_items=train_dataset.num_items,
        emb_dim=emb_dim,
        num_codebooks=num_codebooks,
        num_codewords=num_codewords,
        dropout=dropout,
        **config["model_config"]
    )
    model.to(device)
    if args.eval_only_mode:
        if not args.model_load_path or not args.eval_dataset:
            print("model checkpoint and evaluation dataset must be provided in evaluation only mode.")
        else:
            print("restoring model...")
            model.restore_weights(args.model_load_path)
            print("evaluating...")
            eval_config = {
                "eval_dataset": eval_dataset,
                "device": device,
                "batch_size": config["eval_config"]["batch_size"],
                "num_negatives": eval_dataset.num_negatives
            }
            hr, ndcg = evaluate(model, eval_config)
            print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]))
            print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(
            config['training_config']['learning_rate']), betas=(0.9, 0.98))

        if args.model_load_path:
            model.restore_weights(
                args.model_load_path,
                config["training_config"].get("layers_to_restore", [])
            )

        if "freeze_layers" in config["training_config"]:
            model.freeze_layers(config["training_config"]["freeze_layers"])

        training_config = {
            "train_dataset": train_dataset,
            "negative_sampler": sampler,
            "optimizer": optimizer,
            "device": device,
            "loss_fn": loss_fn,
            "batch_size": config["training_config"]["batch_size"],
            "num_negatives": config["training_config"]["num_negative_samples"],
            "num_epochs": config["training_config"]["num_epochs"],
            "num_workers": config["training_config"]["num_dataloader_workers"],
            "num_init_batches_to_update_ip_table": config["training_config"][
                "initial_num_batches_to_continuously_update_ip_table"],
        }

        train(model, training_config)

        if args.model_save_path:
            model.save_model(args.model_save_path)

        if args.eval_dataset:
            print("evaluating...")
            eval_config = {
                "eval_dataset": eval_dataset,
                "device": device,
                "batch_size": config["eval_config"]["batch_size"],
                "num_negatives": eval_dataset.num_negatives
            }
            hr, ndcg = evaluate(model, eval_config)
            print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]))
            print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]))

            if args.model_save_path:
                with open(args.model_save_path.split('.')[0] + ".txt", 'wt') as f:
                    print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]), file=f)
                    print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]), file=f)