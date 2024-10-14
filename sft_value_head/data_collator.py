import json
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Sequence

from transformers import DataCollatorForSeq2Seq
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from constant import *

def preprocess_value_dataset(examples, tokenizer, cutoff_len):
    model_inputs = {"input_ids": [], "attention_mask": [], "Q": [], "labels": []}

    for i in range(len(examples["instruction"])):
        input_ids = tokenizer.encode(examples["instruction"][i], add_special_tokens=False)
        source_mask = [IGNORE_INDEX] * len(input_ids)
        
        Q = [IGNORE_INDEX] * len(input_ids)
    
        labels = source_mask
        
        multistep_response = examples["output"][i]
        response_state = multistep_response[-1]['Q']  # last Q
        for sub_response in multistep_response:
            sub_Q = float(sub_response['Q'])
            sub_response_ids = tokenizer.encode(sub_response['step'], add_special_tokens=False)
            
            # our value model predicts the v based on '\n'
            input_ids += sub_response_ids
            Q += [IGNORE_INDEX] * (len(sub_response_ids) - 1) + [sub_Q]
            labels += sub_response_ids

            if len(input_ids) > cutoff_len:
                break

        # add eos token
        input_ids += [tokenizer.eos_token_id]
        Q += [IGNORE_INDEX]
        labels += [tokenizer.eos_token_id]

        if len(input_ids) > cutoff_len:
            input_ids = input_ids[:cutoff_len]
            Q = Q[:cutoff_len]
            labels = labels[:cutoff_len]


        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["Q"].append(Q)


        if response_state == -1:
            model_inputs["labels"].append([IGNORE_INDEX] * len(labels))
        elif response_state == 1:
            model_inputs["labels"].append(labels)
        else:
            assert False, response_state

    return model_inputs

@dataclass
class VMDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

            # Padding Q to the longest sequence in the batch
            if "Q" in features[0]:   # float32
                # max_length_Q = max(len(feature["Q"]) for feature in features)
                for feature in features:
                    remainder = [IGNORE_INDEX] * (max_label_length - len(feature["Q"]))  # Assuming IGNORE_INDEX as padding value for Q
                    
                    if isinstance(feature["Q"], list):
                        feature["Q"] = (
                            feature["Q"] + remainder if padding_side == "right" else remainder + feature["Q"]
                        )
                    elif padding_side == "right":
                        feature["Q"] = np.concatenate([feature["Q"], remainder]).astype(np.float32)
                    else:
                        feature["Q"] = np.concatenate([remainder, feature["Q"]]).astype(np.float32)

        features = pad_without_fast_tokenizer_warning(  # only padding input_ids and attention_mask
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

