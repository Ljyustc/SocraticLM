import torch
import numpy as np
from transformers.data.data_collator import DataCollatorForSeq2Seq, pad_without_fast_tokenizer_warning


class Collator(DataCollatorForSeq2Seq):
    def get_text_sequence_effective_length(self, tensor: list):
        # find valid token with max token index in order to skip left padded tokens.
        tensor = torch.tensor(tensor, dtype=torch.long)
        max_token_id_index = torch.max(tensor, dim=-1).indices.item()
        # find first pad token after valid tokens.
        first_pad_token_index = torch.logical_or(
            tensor[max_token_id_index:] == -100,
            tensor[max_token_id_index:] == self.tokenizer.pad_token_id
        ).nonzero(as_tuple=True)[0][0].item()
        max_length = max_token_id_index + first_pad_token_index
        return max_length

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        input_ids_effective_lengths = [self.get_text_sequence_effective_length(feature['input_ids']) for feature in features]
        max_length = max(input_ids_effective_lengths)
        for idx in range(len(features)):
            for key in features[idx]:
                features[idx][key] = features[idx][key][:max_length]

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

        features = pad_without_fast_tokenizer_warning(
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
