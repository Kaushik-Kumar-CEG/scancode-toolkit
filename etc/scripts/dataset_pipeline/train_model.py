# finetunes a DeBERTa token classifier to predict required phrase spans
# reads the BIOES JSONL that build_dataset.py produces
# torch and transformers are imported lazily so the tests can run without them
import inspect
import json
import random
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import click


# BIOES scheme, single source of truth shared by the model and the exporter
LABELS = ['O', 'B-REQ', 'I-REQ', 'E-REQ', 'S-REQ']
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# label id ignored by the loss for padding and non first subwords
IGNORE_INDEX = -100

MODEL_NAME = 'microsoft/deberta-v3-large'
MAX_LENGTH = 512


@dataclass
class Config:
    """Holds the settings for one training run"""
    data_dir: Path
    output_dir: Path
    model_name: str = MODEL_NAME
    max_length: int = MAX_LENGTH

    epochs: int = 8
    batch_size: int = 1
    grad_accum: int = 16
    base_lr: float = 2e-5
    # the classifier and crf start from scratch so they learn at a higher rate
    head_lr: float = 1e-4
    layer_decay: float = 0.98
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.5
    adam_epsilon: float = 1e-6
    early_stopping_patience: int = 3

    # cap examples per split for a quick smoke test, 0 uses everything
    limit: int = 0
    # pick up from the last saved checkpoint in output_dir if a run died
    resume: bool = False
    # average the last few checkpoints into the final model, 0 turns it off
    avg_checkpoints: int = 3

    # turn off to train a plain softmax tagger for ablation
    use_crf: bool = True
    # light weighted cross entropy on top of the CRF loss to protect the rare
    # boundary labels against the O heavy imbalance, set 0 for pure CRF
    aux_ce_weight: float = 0.3
    # report injection success rate too, needs scancode importable
    with_isr: bool = False

    seed: int = 42

    # used only when aux_ce_weight is set, one weight per BIOES label
    label_weights: list = field(default_factory=lambda: [1.0, 2.0, 1.5, 1.5, 2.0])


def set_seed(seed):
    """Seed python, numpy and torch for reproducible runs"""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    """Yield parsed records from a JSONL file"""
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def align_labels(tokens, word_labels, tokenizer, max_length):
    """Tokenize words into subwords and put each label on its first subword

    Continuation subwords and special tokens get IGNORE_INDEX so the loss
    skips them
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    label_ids = []
    previous_word = None
    for word_id in encoding.word_ids():
        if word_id is None:
            label_ids.append(IGNORE_INDEX)
        elif word_id != previous_word:
            label_ids.append(LABEL2ID[word_labels[word_id]])
        else:
            label_ids.append(IGNORE_INDEX)
        previous_word = word_id

    encoding['labels'] = label_ids
    return encoding


class PhraseDataset:
    """Reads a BIOES JSONL split and encodes each rule for the model"""

    def __init__(self, path, tokenizer, max_length, limit=0):
        self.examples = []
        self.truncated = 0
        for record in load_jsonl(path):
            if limit and len(self.examples) >= limit:
                break
            tokens = record['tokens']
            labels = record['bioes_labels']
            if not tokens:
                continue
            encoding = align_labels(tokens, labels, tokenizer, max_length)
            kept = sum(1 for lid in encoding['labels'] if lid != IGNORE_INDEX)
            if kept < len(tokens):
                self.truncated += 1
            self.examples.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': encoding['labels'],
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def extract_spans(tags):
    """Return the set of (start, end) word spans in a BIOES tag sequence

    Tolerates malformed sequences so it also works on raw softmax output
    """
    spans = []
    start = None
    for i, tag in enumerate(tags):
        if tag == 'S-REQ':
            spans.append((i, i))
            start = None
        elif tag == 'B-REQ':
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag == 'I-REQ':
            if start is None:
                start = i
        elif tag == 'E-REQ':
            if start is None:
                start = i
            spans.append((start, i))
            start = None
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(tags) - 1))
    return set(spans)


def decode_row(pred_row, label_row):
    """Drop the ignored positions and map ids back to BIOES tags"""
    pred_tags = []
    true_tags = []
    for pred, label in zip(pred_row, label_row):
        if int(label) == IGNORE_INDEX:
            continue
        true_tags.append(ID2LABEL[int(label)])
        pred_tags.append(ID2LABEL.get(int(pred), 'O'))
    return pred_tags, true_tags


def compute_metrics(eval_pred):
    """Strict entity level micro F1 plus a rule level exact match

    With a single phrase type this is just the span set overlap, so we score
    it directly and skip the seqeval dependency
    """
    predictions, labels = eval_pred
    tp = fp = fn = 0
    exact = 0
    for pred_row, label_row in zip(predictions, labels):
        predicted, actual = decode_row(pred_row, label_row)
        pred_spans = extract_spans(predicted)
        true_spans = extract_spans(actual)
        tp += len(pred_spans & true_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)
        if pred_spans == true_spans:
            exact += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'exact_match': exact / len(predictions) if len(predictions) else 0.0,
    }


def evaluate_isr(records, model, tokenizer, max_length):
    """Fraction of predicted phrases scancode can still locate in the rule text

    A phrase that cannot be located is not injectable, so this is the metric
    that actually matters
    """
    import torch
    from licensedcode.required_phrases import find_phrase_spans_in_text

    device = next(model.parameters()).device
    model.eval()
    total = 0
    injectable = 0
    for record in records:
        tokens = record['tokens']
        text = record['text']
        if not tokens:
            continue
        encoding = align_labels(tokens, record['bioes_labels'], tokenizer, max_length)
        inputs = {
            'input_ids': torch.tensor([encoding['input_ids']], device=device),
            'attention_mask': torch.tensor([encoding['attention_mask']], device=device),
            'labels': torch.tensor([encoding['labels']], device=device),
        }
        with torch.no_grad():
            output = model(**inputs)
        tags, _ = decode_row(output['predictions'][0].tolist(), output['word_labels'][0].tolist())
        for start, end in extract_spans(tags):
            if end >= len(tokens):
                continue
            phrase = ' '.join(tokens[start:end + 1])
            total += 1
            if find_phrase_spans_in_text(text, phrase):
                injectable += 1

    return injectable / total if total else 0.0


def average_checkpoints(output_dir, num):
    """Mean of the weights from the last few epoch checkpoints

    Averaging nearby checkpoints smooths out the noise from the last steps and
    usually gains a little F1 over picking a single one
    """
    import glob
    from safetensors.torch import load_file

    paths = sorted(
        glob.glob(str(output_dir / 'checkpoint-*')),
        key=lambda p: int(p.split('-')[-1]),
    )[-num:]
    if len(paths) < 2:
        return None

    states = [load_file(str(Path(p) / 'model.safetensors')) for p in paths]
    averaged = {}
    for key in states[0]:
        stacked = [state[key].float() for state in states]
        averaged[key] = sum(stacked) / len(stacked)
    return averaged


def run_training(config):
    """Train, evaluate and save the model, callable from a notebook or main()"""
    import torch
    from transformers import AutoTokenizer
    from transformers import DataCollatorForTokenClassification
    from transformers import EarlyStoppingCallback
    from transformers import TrainingArguments

    from phrase_model import PhraseTagger
    from phrase_model import PhraseTrainer
    from phrase_model import build_optimizer

    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    head = 'CRF' if config.use_crf else 'softmax'
    click.echo(f'training {config.model_name} with a {head} head')

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('need a fast tokenizer for word_ids, got a slow one')
    train_ds = PhraseDataset(config.data_dir / 'train.jsonl', tokenizer, config.max_length, config.limit)
    val_ds = PhraseDataset(config.data_dir / 'val.jsonl', tokenizer, config.max_length, config.limit)
    test_ds = PhraseDataset(config.data_dir / 'test.jsonl', tokenizer, config.max_length, config.limit)

    click.echo(f'examples  train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}')
    if train_ds.truncated:
        click.echo(f'note: {train_ds.truncated} train rules were longer than {config.max_length} subwords and got truncated')

    model = PhraseTagger(config)
    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=IGNORE_INDEX)

    # deberta breaks in fp16, so use bf16 only where the gpu supports it else fp32
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.base_lr,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type='cosine',
        max_grad_norm=config.max_grad_norm,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=max(3, config.avg_checkpoints),
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        bf16=use_bf16,
        fp16=False,
        logging_steps=50,
        report_to='none',
        seed=config.seed,
        dataloader_num_workers=2,
    )

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        optimizers=(build_optimizer(config, model), None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    # transformers renamed the tokenizer arg to processing_class, support both
    if 'processing_class' in inspect.signature(PhraseTrainer.__init__).parameters:
        trainer_kwargs['processing_class'] = tokenizer
    else:
        trainer_kwargs['tokenizer'] = tokenizer

    trainer = PhraseTrainer(**trainer_kwargs)

    trainer.train(resume_from_checkpoint=config.resume or None)

    if config.avg_checkpoints:
        averaged = average_checkpoints(config.output_dir, config.avg_checkpoints)
        if averaged:
            model.load_state_dict(averaged, strict=False)
            click.echo(f'averaged the last {config.avg_checkpoints} checkpoints')

    trainer.save_model(str(config.output_dir))

    # the exporter reads this back to rebuild the model
    train_config = config.output_dir / 'train_config.json'
    train_config.write_text(json.dumps({
        'model_name': config.model_name,
        'use_crf': config.use_crf,
        'max_length': config.max_length,
        'labels': LABELS,
    }, indent=2))

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix='test')
    click.echo(f'test: {test_metrics}')

    if config.with_isr:
        records = list(load_jsonl(config.data_dir / 'test.jsonl'))
        isr = evaluate_isr(records, model, tokenizer, config.max_length)
        click.echo(f'injection success rate: {isr:.4f}')
        test_metrics['test_isr'] = isr

    return test_metrics


@click.command()
@click.option('--data-dir', required=True, type=click.Path(exists=True),
              help='Directory holding train.jsonl, val.jsonl and test.jsonl')
@click.option('--output-dir', default='model-output',
              help='Where to write the trained model and metrics')
@click.option('--model-name', default=MODEL_NAME, help='Base model to finetune')
@click.option('--epochs', default=8, type=int)
@click.option('--base-lr', default=2e-5, type=float, help='Learning rate for the backbone')
@click.option('--head-lr', default=1e-4, type=float, help='Learning rate for the classifier and crf head')
@click.option('--aux-ce-weight', default=0.3, type=float,
              help='Weight of the auxiliary weighted cross entropy, 0 for pure CRF')
@click.option('--no-crf', is_flag=True, default=False,
              help='Train a plain softmax tagger instead of the CRF head')
@click.option('--with-isr', is_flag=True, default=False,
              help='Also report injection success rate, needs scancode installed')
@click.option('--limit', default=0, type=int,
              help='Cap examples per split for a quick smoke test, 0 uses everything')
@click.option('--resume', is_flag=True, default=False,
              help='Resume from the last checkpoint in output-dir after a crash')
@click.option('--avg-checkpoints', default=3, type=int,
              help='Average the last N checkpoints into the final model, 0 turns it off')
@click.option('--seed', default=42, type=int)
def main(data_dir, output_dir, model_name, epochs, base_lr, head_lr, aux_ce_weight,
         no_crf, with_isr, limit, resume, avg_checkpoints, seed):
    """Train the required phrase tagger from a BIOES dataset"""
    config = Config(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        model_name=model_name,
        epochs=epochs,
        base_lr=base_lr,
        head_lr=head_lr,
        aux_ce_weight=aux_ce_weight,
        use_crf=not no_crf,
        with_isr=with_isr,
        limit=limit,
        resume=resume,
        avg_checkpoints=avg_checkpoints,
        seed=seed,
    )
    run_training(config)


if __name__ == '__main__':
    main()
