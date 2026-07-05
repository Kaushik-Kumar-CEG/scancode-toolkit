# exports a trained tagger to ONNX so it can run on CPU without torch
#
# the CRF layer cannot be put into ONNX, so we export only the backbone and
# the classifier (the emission scores) and save the learned CRF transition
# matrix beside it. at inference a small numpy viterbi decoder reproduces the
# exact CRF decoding from those transitions
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np


def viterbi_decode(emissions, start_transitions, transitions, end_transitions):
    """Best valid tag path for one sequence, matches pytorch-crf decoding

    emissions is (seq_len, num_tags), transitions[i, j] scores tag i to tag j
    """
    seq_len = emissions.shape[0]
    score = start_transitions + emissions[0]
    backpointers = []
    for step in range(1, seq_len):
        candidates = score[:, None] + transitions
        best_source = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + emissions[step]
        backpointers.append(best_source)

    score = score + end_transitions
    best = int(score.argmax())
    path = [best]
    for sources in reversed(backpointers):
        best = int(sources[best])
        path.append(best)
    path.reverse()
    return path


def sha256(path):
    """Hex sha256 of a file"""
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def build_emissions_module(tagger):
    """Wrap the backbone and classifier as a plain emissions only model"""
    import torch.nn as nn

    class EmissionsModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = tagger.backbone
            self.classifier = tagger.classifier

        def forward(self, input_ids, attention_mask):
            hidden = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            return self.classifier(hidden)

    return EmissionsModule().eval()


def load_tagger(model_dir, train_config):
    """Rebuild the tagger and load the finetuned weights"""
    import torch
    from phrase_model import PhraseTagger

    cfg = SimpleNamespace(
        model_name=train_config['model_name'],
        use_crf=train_config['use_crf'],
        aux_ce_weight=0.0,
        label_weights=[1.0] * len(train_config['labels']),
    )
    tagger = PhraseTagger(cfg)

    safetensors_file = Path(model_dir) / 'model.safetensors'
    if safetensors_file.exists():
        from safetensors.torch import load_file
        state = load_file(str(safetensors_file))
    else:
        state = torch.load(Path(model_dir) / 'pytorch_model.bin', map_location='cpu')
    # class_weights is a training only buffer, it may be absent here
    state = {k: v for k, v in state.items() if k != 'class_weights'}
    tagger.load_state_dict(state)
    return tagger.eval()


def check_viterbi_matches_crf(tagger, num_tags):
    """Random emissions must decode the same in numpy and in pytorch-crf"""
    import torch

    start = tagger.crf.start_transitions.detach().cpu().numpy()
    transitions = tagger.crf.transitions.detach().cpu().numpy()
    end = tagger.crf.end_transitions.detach().cpu().numpy()

    emissions = torch.randn(3, 14, num_tags)
    mask = torch.ones(3, 14, dtype=torch.bool)
    crf_paths = tagger.crf.decode(emissions, mask=mask)
    for row in range(emissions.size(0)):
        numpy_path = viterbi_decode(emissions[row].numpy(), start, transitions, end)
        if numpy_path != crf_paths[row]:
            raise AssertionError('numpy viterbi disagrees with pytorch-crf decoding')

    return start, transitions, end


def export(model_dir, output_dir, opset):
    """Export to ONNX, dump CRF transitions and write a checksum manifest"""
    import torch
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = json.loads((model_dir / 'train_config.json').read_text())
    labels = train_config['labels']
    use_crf = train_config['use_crf']

    tagger = load_tagger(model_dir, train_config)
    emissions_module = build_emissions_module(tagger)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    sample = tokenizer(
        'Licensed under the Apache License Version 2.0',
        return_tensors='pt',
    )
    inputs = (sample['input_ids'], sample['attention_mask'])

    onnx_path = output_dir / 'model.onnx'
    torch.onnx.export(
        emissions_module,
        inputs,
        str(onnx_path),
        input_names=['input_ids', 'attention_mask'],
        output_names=['emissions'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'emissions': {0: 'batch', 1: 'sequence'},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    click.echo(f'wrote {onnx_path}')

    manifest = {'labels': labels, 'onnx_model': sha256(onnx_path)}

    if use_crf:
        start, transitions, end = check_viterbi_matches_crf(tagger, len(labels))
        transitions_path = output_dir / 'crf_transitions.npz'
        np.savez(transitions_path, start=start, transitions=transitions, end=end)
        manifest['crf_transitions'] = sha256(transitions_path)
        click.echo(f'wrote {transitions_path}  (viterbi matches pytorch-crf)')

    # the onnx graph must produce the same emissions as pytorch
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    feeds = {
        'input_ids': sample['input_ids'].numpy(),
        'attention_mask': sample['attention_mask'].numpy(),
    }
    onnx_emissions = session.run(['emissions'], feeds)[0]
    torch_emissions = emissions_module(*inputs).detach().numpy()
    if not np.allclose(onnx_emissions, torch_emissions, atol=1e-3):
        raise AssertionError('onnx emissions differ from pytorch emissions')
    click.echo('onnx emissions match pytorch within tolerance')

    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    click.echo(f'wrote {manifest_path}')


@click.command()
@click.option('--model-dir', required=True, type=click.Path(exists=True),
              help='Directory with the trained model and train_config.json')
@click.option('--output-dir', default=None,
              help='Where to write the ONNX files, defaults to the model dir')
@click.option('--opset', default=14, type=int, help='ONNX opset version')
def main(model_dir, output_dir, opset):
    """Export a trained required phrase tagger to ONNX"""
    export(model_dir, output_dir or model_dir, opset)


if __name__ == '__main__':
    main()
