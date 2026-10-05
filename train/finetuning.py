"""Fine-tune a pre-trained PULSAR checkpoint on new data, using the same
self-supervised masked-reconstruction objective as pre-training.

Example:
    python train/finetuning.py \\
        --model_name_or_path KuanP/PULSAR-pbmc \\
        --train_h5ad my_data.h5ad \\
        --output_dir outputs/pulsar-finetuned

See ``pretraining.py`` for the input format and the shared arguments.
"""

from pretraining import build_parser, masking_config, train
from pulsar.model import PULSARForSelfSupervisedFinetuning


def main():
    parser = build_parser("Self-supervised fine-tuning of a pre-trained PULSAR model")
    parser.add_argument("--model_name_or_path", required=True, help="Pre-trained checkpoint (local path or HuggingFace model ID)")
    parser.set_defaults(
        num_train_epochs=2048,
        per_device_train_batch_size=32,
        weight_decay=0.01,
        warmup_steps=500,
        eval_steps=1000,
    )
    args = parser.parse_args()

    model = PULSARForSelfSupervisedFinetuning.from_pretrained(
        args.model_name_or_path, **masking_config(args)
    )
    train(model, args)


if __name__ == "__main__":
    main()
