import logging
import os
import time

import torch
from matplotlib import pyplot as plt
from torch import optim as optim
from torch.nn.utils import clip_grad_norm_

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from src.misc import load_pickle, save_as_pickle
from src.model.ALBERT.modeling_albert import AlbertModel
from src.preprocessing_funcs import load_dataloaders
from src.train_funcs import (
    Two_Headed_Loss,
    evaluate_,
    load_results,
    load_state,
)

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL,
)
logger = logging.getLogger(__file__)


class Pretrainer:
    def __init__(self, config):
        self.gradient_acc_steps = config.get("gradient_acc_steps")
        self.config = config
        self.data_loader = load_dataloaders(self.config)
        self.train_len = len(self.data_loader)

    def train(self):
        train_on_gpu = torch.cuda.is_available()

        logger.info("Loaded %d pre-training samples." % self.train_len)

        model_name = "ALBERT"
        net = AlbertModel.from_pretrained(
            "albert-large-v2",
            force_download=False,
            model_size="albert-large-v2",
        )

        tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
        net.resize_token_embeddings(len(tokenizer))
        e1_id = tokenizer.convert_tokens_to_ids("[E1]")
        e2_id = tokenizer.convert_tokens_to_ids("[E2]")
        assert e1_id != e2_id != 1

        if train_on_gpu:
            net.cuda()

        criterion = Two_Headed_Loss(
            lm_ignore_idx=tokenizer.pad_token_id,
            use_logits=True,
            normalize=False,
        )
        optimizer = optim.Adam(
            [{"params": net.parameters(), "lr": self.config.get("lr")}]
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[2, 4, 6, 8, 12, 15, 18, 20, 22, 24, 26, 30],
            gamma=0.8,
        )

        start_epoch, best_pred, amp_checkpoint = load_state(
            net, optimizer, scheduler, self.config, load_best=False
        )

        losses_per_epoch, accuracy_per_epoch = load_results(1)

        logger.info("Starting training process...")
        pad_id = tokenizer.pad_token_id
        mask_id = tokenizer.mask_token_id
        update_size = len(self.data_loader) // 10
        for epoch in range(start_epoch, self.config.get("epochs")):
            start_time = time.time()
            net.train()
            total_loss = 0.0
            losses_per_batch = []
            total_acc = 0.0
            lm_accuracy_per_batch = []
            for i, data in enumerate(self.data_loader, 0):
                (
                    x,
                    masked_for_pred,
                    e1_e2_start,
                    _,
                    blank_labels,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = data
                masked_for_pred = masked_for_pred[(masked_for_pred != pad_id)]
                if masked_for_pred.shape[0] == 0:
                    print("Empty dataset, skipping...")
                    continue
                attention_mask = (x != pad_id).float()
                token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

                if train_on_gpu:
                    x = x.cuda()
                    masked_for_pred = masked_for_pred.cuda()
                    attention_mask = attention_mask.cuda()
                    token_type_ids = token_type_ids.cuda()

                blanks_logits, lm_logits = net(
                    x,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    Q=None,
                    e1_e2_start=e1_e2_start,
                )
                lm_logits = lm_logits[(x == mask_id)]

                if (i % update_size) == (update_size - 1):
                    verbose = True
                else:
                    verbose = False

                loss = criterion(
                    lm_logits,
                    blanks_logits,
                    masked_for_pred,
                    blank_labels,
                    verbose=verbose,
                )
                loss = loss / self.gradient_acc_steps

                loss.backward()

                clip_grad_norm_(net.parameters(), self.config.get("max_norm"))

                if (i % self.gradient_acc_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                total_acc += evaluate_(
                    lm_logits,
                    blanks_logits,
                    masked_for_pred,
                    blank_labels,
                    tokenizer,
                    print_=False,
                )[0]

                if (i % update_size) == (update_size - 1):
                    losses_per_batch.append(
                        self.gradient_acc_steps * total_loss / update_size
                    )
                    lm_accuracy_per_batch.append(total_acc / update_size)
                    print(
                        "[Epoch: %d, %5d/ %d points] total loss, lm accuracy per batch: %.3f, %.3f"
                        % (
                            epoch + 1,
                            (i + 1),
                            self.train_len,
                            losses_per_batch[-1],
                            lm_accuracy_per_batch[-1],
                        )
                    )
                    total_loss = 0.0
                    total_acc = 0.0
                    logger.info(
                        "Last batch samples (pos, neg): %d, %d"
                        % (
                            (blank_labels.squeeze() == 1).sum().item(),
                            (blank_labels.squeeze() == 0).sum().item(),
                        )
                    )

            scheduler.step()
            losses_per_epoch.append(
                sum(losses_per_batch) / len(losses_per_batch)
            )
            accuracy_per_epoch.append(
                sum(lm_accuracy_per_batch) / len(lm_accuracy_per_batch)
            )
            print(
                "Epoch finished, took %.2f seconds."
                % (time.time() - start_time)
            )
            print(
                "Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1])
            )
            print(
                "Accuracy at Epoch %d: %.7f"
                % (epoch + 1, accuracy_per_epoch[-1])
            )

            if accuracy_per_epoch[-1] > best_pred:
                best_pred = accuracy_per_epoch[-1]
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": net.state_dict(),
                        "best_acc": accuracy_per_epoch[-1],
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    os.path.join("./data/", "test_model_best_ALBERT.pth.tar"),
                )

            if (epoch % 1) == 0:
                save_as_pickle(
                    "test_losses_per_epoch_ALBERT.pkl", losses_per_epoch,
                )
                save_as_pickle(
                    "test_accuracy_per_epoch_ALBERT.pkl", accuracy_per_epoch,
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": net.state_dict(),
                        "best_acc": accuracy_per_epoch[-1],
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    os.path.join("./data/", "test_checkpoint_ALBERT.pth.tar"),
                )

        logger.info("Finished Training!")
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        ax.scatter([e for e in range(len(losses_per_epoch))], losses_per_epoch)
        ax.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax.set_xlabel("Epoch", fontsize=22)
        ax.set_ylabel("Training Loss per batch", fontsize=22)
        ax.set_title("Training Loss vs Epoch", fontsize=32)
        plt.savefig(os.path.join("./data/", "loss_vs_epoch_ALBERT.png"))

        fig2 = plt.figure(figsize=(20, 20))
        ax2 = fig2.add_subplot(111)
        ax2.scatter(
            [e for e in range(len(accuracy_per_epoch))], accuracy_per_epoch
        )
        ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax2.set_xlabel("Epoch", fontsize=22)
        ax2.set_ylabel("Test Masked LM Accuracy", fontsize=22)
        ax2.set_title("Test Masked LM Accuracy vs Epoch", fontsize=32)
        plt.savefig(os.path.join("./data/", "accuracy_vs_epoch_ALBERT.png"))

        return net
