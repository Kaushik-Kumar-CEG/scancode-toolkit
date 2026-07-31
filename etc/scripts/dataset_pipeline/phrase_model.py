# the DeBERTa token classifier used to tag required phrases
# kept in its own module so train_model.py stays importable without torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchcrf import CRF
from transformers import AutoModel
from transformers import Trainer

from train_model import first_subword_positions
from train_model import IGNORE_INDEX
from train_model import LABELS


class PhraseTagger(nn.Module):
    """DeBERTa backbone with a token classifier and an optional CRF head

    The CRF scores the first subword of each word, so tagging happens at word
    level and it decodes the best valid label sequence per rule
    """

    def __init__(self, config):
        super().__init__()
        self.use_crf = config.use_crf
        self.aux_ce_weight = config.aux_ce_weight
        self.num_labels = len(LABELS)

        # newer transformers loads the checkpoint in its stored fp16, force fp32
        # so the fresh classifier head matches and deberta stays numerically stable
        self.backbone = AutoModel.from_pretrained(config.model_name).float()
        self.backbone.gradient_checkpointing_enable()
        # without this the checkpointed backbone can get no gradient and only the
        # head trains, while the loss still looks fine
        self.backbone.enable_input_require_grads()

        hidden_size = self.backbone.config.hidden_size
        dropout = getattr(self.backbone.config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=True)

        if self.aux_ce_weight > 0:
            self.register_buffer(
                'class_weights',
                torch.tensor(config.label_weights, dtype=torch.float),
            )
        else:
            self.class_weights = None

    def emissions(self, input_ids, attention_mask):
        """Per subword label scores from the backbone"""
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return self.classifier(self.dropout(hidden))

    def token_cross_entropy(self, emissions, labels):
        """Plain weighted cross entropy over subwords, ignores -100"""
        return F.cross_entropy(
            emissions.reshape(-1, self.num_labels),
            labels.reshape(-1),
            weight=self.class_weights,
            ignore_index=IGNORE_INDEX,
        )

    def gather_words(self, emissions, labels):
        """Pack the first subword of each word into a dense left aligned sequence

        crf_tags pad with 0 so the CRF has valid indices, eval_tags pad with
        IGNORE_INDEX so the metrics skip them
        """
        batch, _, num_labels = emissions.shape
        is_word = labels.ne(IGNORE_INDEX)
        lengths = is_word.sum(dim=1)
        width = int(lengths.max().item())

        word_emissions = emissions.new_zeros((batch, width, num_labels))
        crf_tags = labels.new_zeros((batch, width))
        eval_tags = labels.new_full((batch, width), IGNORE_INDEX)
        mask = torch.zeros((batch, width), dtype=torch.bool, device=emissions.device)

        for row in range(batch):
            positions = is_word[row].nonzero(as_tuple=True)[0]
            count = positions.numel()
            word_emissions[row, :count] = emissions[row, positions]
            tags = labels[row, positions]
            crf_tags[row, :count] = tags
            eval_tags[row, :count] = tags
            mask[row, :count] = True

        return word_emissions, crf_tags, eval_tags, mask

    def forward(self, input_ids, attention_mask, labels=None):
        emissions = self.emissions(input_ids, attention_mask)
        result = {}

        if not self.use_crf:
            if labels is not None:
                result['loss'] = self.token_cross_entropy(emissions, labels)
                result['word_labels'] = labels
            if not self.training:
                result['predictions'] = emissions.argmax(dim=-1)
            return result

        if labels is None:
            raise ValueError('CRF head needs labels to locate words, use the ONNX export for inference')

        word_emissions, crf_tags, eval_tags, mask = self.gather_words(emissions, labels)
        # the CRF math is more stable in fp32 under mixed precision
        word_emissions = word_emissions.float()

        log_likelihood = self.crf(word_emissions, crf_tags, mask=mask, reduction='mean')
        loss = -log_likelihood
        if self.aux_ce_weight > 0:
            loss = loss + self.aux_ce_weight * self.token_cross_entropy(emissions, labels)
        result['loss'] = loss
        result['word_labels'] = eval_tags

        if not self.training:
            decoded = self.crf.decode(word_emissions, mask=mask)
            result['predictions'] = self.pad_decoded(decoded, mask.size(1), emissions.device)

        return result

    def predict_words(self, input_ids, attention_mask, word_ids):
        """Label id per word for a single rule, without labels

        forward() needs labels to find the first subword of each word, which we
        do not have at inference time, so take those positions from the
        tokenizer word_ids and decode from the CRF directly
        """
        emissions = self.emissions(input_ids, attention_mask)
        positions = first_subword_positions(word_ids)
        if not positions:
            return []

        word_emissions = emissions[0, positions].unsqueeze(0).float()
        if not self.use_crf:
            return word_emissions.argmax(dim=-1)[0].tolist()

        mask = torch.ones(word_emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        return self.crf.decode(word_emissions, mask=mask)[0]

    def pad_decoded(self, decoded, width, device):
        """Turn the variable length CRF paths into a padded tensor"""
        preds = torch.full((len(decoded), width), IGNORE_INDEX, dtype=torch.long, device=device)
        for row, path in enumerate(decoded):
            if path:
                preds[row, :len(path)] = torch.tensor(path, dtype=torch.long, device=device)
        return preds


def build_optimizer(config, model):
    """AdamW with layer wise learning rate decay

    The head learns fastest and the embeddings slowest so the pretrained lower
    layers keep more of what they know
    """
    num_layers = model.backbone.config.num_hidden_layers
    no_decay = ('bias', 'LayerNorm.weight', 'layer_norm.weight')

    def rate_for(name):
        if name.startswith('classifier') or name.startswith('crf'):
            return config.head_lr
        if '.encoder.layer.' in name:
            layer = int(name.split('.encoder.layer.')[1].split('.')[0])
            return config.base_lr * (config.layer_decay ** (num_layers - layer))
        # embeddings and relative position embeddings sit at the bottom
        return config.base_lr * (config.layer_decay ** (num_layers + 1))

    groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        decay = 0.0 if any(nd in name for nd in no_decay) else config.weight_decay
        groups.append({'params': [param], 'lr': rate_for(name), 'weight_decay': decay})

    # the 8 bit optimizer keeps deberta-large inside a 16gb gpu, fall back to
    # plain AdamW where bitsandbytes is not installed
    try:
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(groups, lr=config.base_lr, eps=config.adam_epsilon, betas=(0.9, 0.999))
    except ImportError:
        return AdamW(groups, lr=config.base_lr, eps=config.adam_epsilon, betas=(0.9, 0.999))


class PhraseTrainer(Trainer):
    """Trainer that reads loss and predictions from our model output dict"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.get('loss')
            if loss is not None:
                loss = loss.detach()
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, outputs['predictions'], outputs['word_labels'])
