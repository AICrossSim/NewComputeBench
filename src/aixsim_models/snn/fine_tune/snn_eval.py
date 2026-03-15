from aixsim_models.snn.neuron.st_bifnode import ST_BIFNode
from aixsim_models.snn.snn_layers import (
    RobertaSelfAttentionZIPTF,
    SoftmaxZIPTF,
    LayerNormZIPTF,
    EmbeddingZIPTF,
    LinearUnfoldBias
)

import torch.nn as nn 
import torch
import time
from tqdm import tqdm


def get_subtensors(tensor, mean, std, sample_grain=255, output_num=4):
    for i in range(int(sample_grain)):
        output = (tensor / sample_grain).unsqueeze(0)
        # output = (tensor).unsqueeze(0)
        if i == 0:
            accu = output
        else:
            accu = torch.cat((accu, output), dim=0)
    return accu


def reset_model(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if (
            isinstance(child, ST_BIFNode)
            or isinstance(child, LinearUnfoldBias)
            or isinstance(child, RobertaSelfAttentionZIPTF)
            or isinstance(child, LayerNormZIPTF)
            or isinstance(child, EmbeddingZIPTF)
        ):
            model._modules[name].reset()
            is_need = True
        if not is_need:
            reset_model(child)


class Judger:
    def __init__(self):
        self.network_finish = True

    def judge_finish(self, model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if (
                isinstance(child, ST_BIFNode)
                or isinstance(child, LinearUnfoldBias)
                # or isinstance(child, LLConv2d)
            ):
                self.network_finish = self.network_finish and (
                    not model._modules[name].is_work
                )
                # print("child",child,"network_finish",self.network_finish,"model._modules[name].is_work",(model._modules[name].is_work))
                is_need = True
            if not is_need:
                self.judge_finish(child)

    def reset_network_finish_flag(self):
        self.network_finish = True

# NOTE: This SNNWrapper is not used in the evaluation and fine-tuning process.
# NOTE: We report the results of quantized models before converting them to SNNs.
# NOTE: The accuracy of the SNN models should be the same as the quantized models given enough time steps than matches the bit level.
class MASE_SNNWrapper(nn.Module):

    def __init__(self, ann_model, cfg, time_step=2000, Encoding_type="rate", **kwargs):
        super(MASE_SNNWrapper, self).__init__()
        self.T = time_step
        self.cfg = cfg
        self.finish_judger = Judger()
        self.Encoding_type = Encoding_type
        self.level = kwargs["level"]
        self.neuron_type = kwargs["neuron_type"]
        self.model = ann_model
        self.kwargs = kwargs
        self.model_name = kwargs["model_name"]
        self.is_regression = kwargs.get("is_regression", False)
        self.max_T = 0
        self.model_reset = None
        # self._replace_weight(self.model)
        # self.model_reset = deepcopy(self.model)

    def reset(self):
        # self.model = deepcopy(self.model_reset).cuda()
        reset_model(self)

    def forward(self, input_ids, attention_mask, segments=None, labels=None, verbose=False):
        seqs = input_ids
        masks = attention_mask
        accu = None
        count1 = 0
        accu_per_timestep = []
        # print("self.bit",self.bit)
        # x = x*(2**self.bit-1)+0.0

        if self.Encoding_type == "rate":
            self.mean = 0.0
            self.std = 0.0
            seqs = get_subtensors(seqs, self.mean, self.std, sample_grain=self.level)
            # print("x.shape",x.shape)
        while 1:
            self.finish_judger.reset_network_finish_flag()
            self.finish_judger.judge_finish(self)
            network_finish = self.finish_judger.network_finish
            # print(f"==================={count1}===================")
            if (count1 > 0 and network_finish) or count1 >= self.T:
                self.max_T = max(count1, self.max_T)
                break

            if self.Encoding_type == "rate":
                if count1 < seqs.shape[0]:
                    seqs = seqs[count1]
                else:
                    seqs = torch.zeros(seqs[0].shape).to(seqs.device)
            else:
                if count1 == 0:
                    seqs = seqs
                else:
                    seqs = torch.zeros(seqs.shape).to(seqs.device)
            # elif self.neuron_type == 'IF':
            #     input = x
            # else:
            #     print("No implementation of neuron type:",self.neuron_type)
            #     sys.exit(0)
            # output = self.model(input_ids=seqs, attention_mask=masks, token_type_ids=segments, labels=labels)[1]
            outputs = self.model(input_ids=seqs, attention_mask=masks, token_type_ids=segments, labels=labels)
            if not self.is_regression:
                predictions = outputs.logits.argmax(dim=-1)
            else:
                predictions = outputs.logits.squeeze()

            # print(count1,output[0,0:100])
            # print(count1,"output",torch.abs(output.sum()))

            if count1 == 0:
                accu = predictions + 0.0
            else:
                accu = accu + predictions
            if verbose:
                accu_per_timestep.append(accu)
            # print("accu",accu.sum(),"output",output.sum())
            count1 = count1 + 1
            if count1 % 100 == 0:
                print(count1)

        # print("verbose",verbose)
        breakpoint()
        if verbose:
            accu_per_timestep = torch.stack(accu_per_timestep, dim=0)
            return accu, count1, accu_per_timestep
        else:
            return accu, count1


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def validate(model, dataloader, mode, target_dir):
    """
    Compute the loss and accuracy of a model on some validation dataset.
    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
        roc_auc_score(all_labels, all_prob): The auc computed on the entire validation set.
        all_prob: The probability of classification as label 1 on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    correct_per_timestep_all = []
    total_num = 0
    max_T = 0
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        # for batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels in tqdm(
        #     dataloader
        # ):
        for step, batch in enumerate(dataloader):
            # Move input and output data to the GPU if one is used.
            seqs = batch["input_ids"].to(device)
            total_num = total_num + seqs.shape[0]
            masks = batch["attention_mask"].to(device)
            segments = batch.get("token_type_ids", None)
            if segments is not None:
                segments = segments.to(device)
            labels = batch["labels"].to(device)
            if mode == "SNN":
                output, count, accu_per_timestep = model(seqs, masks, segments, labels, verbose=True)
                # print(accu_per_timestep.shape, count)
                max_T = max(max_T, count)
                print(max_T)
                breakpoint()
                if accu_per_timestep.shape[0] < max_T:
                    padding_per_timestep = accu_per_timestep[-1].unsqueeze(0)
                    padding_length = max_T - accu_per_timestep.shape[0]
                    accu_per_timestep = torch.cat(
                        [
                            accu_per_timestep,
                            padding_per_timestep.repeat(padding_length, 1, 1),
                        ],
                        dim=0,
                    )

                _, predicted_per_time_step = torch.max(accu_per_timestep.data, 2)
                correct_per_timestep = torch.sum((predicted_per_time_step == labels.unsqueeze(0)), dim=1)

                # statistic the total correct sample per time-step
                for i in range(len(correct_per_timestep)):
                    if i >= len(correct_per_timestep_all):
                        if i == 0:
                            correct_per_timestep_all.append(0)
                        else:
                            correct_per_timestep_all.append(
                                correct_per_timestep_all[-1]
                            )

                for i in range(len(correct_per_timestep)):
                    correct_per_timestep_all[i] = (
                        correct_per_timestep_all[i] + correct_per_timestep[i]
                    )

                loss = criterion(output, labels)
                probabilities = nn.functional.softmax(output)
                running_loss += loss.item()
                running_accuracy += correct_predictions(probabilities, labels)
                all_prob.extend(probabilities[:, 1].cpu().numpy())
                all_labels.extend(labels)
                acc_per_timestep_all = []
                for i in range(len(correct_per_timestep_all)):
                    acc_per_timestep_all.append(
                        correct_per_timestep_all[i] / total_num * 100
                    )
                print("mid: acc_per_timestep", acc_per_timestep_all)
                print("mid: running_accuracy", running_accuracy / total_num * 100)
                model.reset()
            else:
                loss, logits, probabilities = model(seqs, masks, segments, labels)
                running_loss += loss.item()
                running_accuracy += correct_predictions(probabilities, labels)
                all_prob.extend(probabilities[:, 1].cpu().numpy())
                all_labels.extend(labels)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    f = open(f"{target_dir}/accuracy_per_timestep.txt", "w+")
    for i in range(len(correct_per_timestep_all)):
        f.write(
            f"T={i}, acc={correct_per_timestep_all[i]/len(dataloader.dataset) * 100}%\n"
        )
    f.close()
    return epoch_time, epoch_loss, epoch_accuracy, all_prob

