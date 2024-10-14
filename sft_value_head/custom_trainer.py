import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from trl import SFTTrainer

from constant import *


class CustomTrainer(SFTTrainer):
    def __init__(self, weight_alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.weight_alpha = weight_alpha
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        # Compute rewards
        Q = inputs.get("Q", None)
        if Q is not None:
            del inputs["Q"]
        
        labels = inputs.get("labels", None)
        if labels is not None:
            del inputs["labels"]
            
        mask = Q.ne(IGNORE_INDEX)

        lm_logits, loss, values = model(**inputs, output_hidden_states=True, return_dict=True)
        values = torch.tanh(values)

        if loss is None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if torch.all(shift_labels==IGNORE_INDEX):
                loss_fct = CrossEntropyLoss(reduction='sum')
            else:
                loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, model.pretrained_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        assert not torch.isnan(loss) and Q is not None

        Q = Q.type_as(values)
        masked_values = torch.where(mask, values, Q)
        value_loss = F.mse_loss(masked_values, Q, reduction='sum') / (mask.sum() + 1e-3)
        all_losses = loss + self.weight_alpha * value_loss

        # if return_outputs:
        #     return all_losses, [all_losses, loss, value_loss, masked_values, Q]
        # return all_losses
        return value_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """should return a tuple of (loss, logits, labels)"""
        # Compute rewards
        Q = inputs.get("Q", None)
        if Q is not None:
            del inputs["Q"]
        
        labels = inputs.get("labels", None)
        if labels is not None:
            del inputs["labels"] 
        mask = Q.ne(IGNORE_INDEX)

        with torch.no_grad():
            lm_logits, loss, values = model(**inputs, output_hidden_states=True, return_dict=True)
            values = torch.tanh(values)

            assert Q is not None

            Q = Q.type_as(values)
            masked_values = torch.where(mask, values, Q)
            value_loss = F.mse_loss(masked_values, Q, reduction='sum') / (mask.sum() + 1e-3)

        return value_loss, None, None