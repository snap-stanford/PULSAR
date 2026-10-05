"""Pre-train PULSAR from scratch with masked reconstruction of cell embeddings.

The input is an H5AD file with pre-computed cell embeddings (e.g. UCE) in
``adata.obsm`` and a donor ID column in ``adata.obs``. By default, 50% of the
sampled cells of each donor are encoded and 85% of the embedding dimensions of
the other cells are masked and reconstructed. Pass
``--coordinated_masking_program_path`` to mask groups of correlated dimensions
together instead of independent dimensions.

Example:
    python train/pretraining.py \\
        --train_h5ad train.h5ad \\
        --valid_h5ad valid.h5ad \\
        --output_dir outputs/pulsar-pretrained

All visible GPUs are used, so the number of donors per step is
``per_device_train_batch_size`` times the number of GPUs.
"""

import argparse
from functools import partial

import anndata as ad
from transformers import Trainer, TrainingArguments

from pulsar.dataset import DonorDatasetAnndata
from pulsar.model import PULSARConfig, PULSARForSelfSupervisedPretraining
from pulsar.utils import collate_fn


def build_parser(description: str) -> argparse.ArgumentParser:
    """Build the argument parser shared by pre-training and fine-tuning."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument("--train_h5ad", required=True, help="H5AD file with the training donors")
    parser.add_argument("--valid_h5ad", default=None, help="H5AD file with the validation donors")
    parser.add_argument("--donor_id_key", default="donor_id", help="Column in adata.obs with donor IDs")
    parser.add_argument("--embedding_key", default="X_uce", help="Key in adata.obsm with cell embeddings")
    parser.add_argument("--max_length", type=int, default=1024, help="Number of cells sampled per donor")
    parser.add_argument("--output_dir", required=True, help="Where to save checkpoints and the final model")

    # Masking
    parser.add_argument("--encoder_ratio", type=float, default=0.5, help="Fraction of cells fed to the encoder")
    parser.add_argument("--element_mask_ratio", type=float, default=0.85, help="Fraction of embedding dimensions masked in the other cells")
    parser.add_argument(
        "--coordinated_masking_program_path",
        default=None,
        help="CSV with columns Dimension and Cluster. If set, whole clusters of correlated dimensions are masked together",
    )

    # Optimization
    parser.add_argument("--num_train_epochs", type=int, default=4096, help="Passes over the donors, each with newly sampled cells")
    parser.add_argument("--per_device_train_batch_size", type=int, default=96, help="Donors per GPU in each step")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Peak learning rate of the cosine schedule")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Linear warmup steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate and save a checkpoint every N steps")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="Data loading worker processes")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision")
    parser.add_argument("--report_to", default="none", help="Logging integration, e.g. wandb")
    return parser


def masking_config(args: argparse.Namespace) -> dict:
    """Masking settings stored in the model config."""
    correlated = args.coordinated_masking_program_path is not None
    return {
        "encoder_ratio": args.encoder_ratio,
        "element_mask_ratio": args.element_mask_ratio,
        "element_mask_strategy": "correlated_program" if correlated else "dimension",
        "coordinated_masking_program_path": args.coordinated_masking_program_path,
    }


def collate_cells(batch: list, max_length: int) -> dict:
    """Sample ``max_length`` cells per donor and keep only the model input."""
    return {"cell_embedding": collate_fn(batch, max_length)["cell_embedding"].float()}


def load_dataset(h5ad_path: str, args: argparse.Namespace) -> DonorDatasetAnndata:
    """Load an H5AD file as a donor-level dataset."""
    return DonorDatasetAnndata(
        ad.read_h5ad(h5ad_path),
        donor_id_key=args.donor_id_key,
        embedding_key=args.embedding_key,
        device="cpu",
    )


def train(model: PULSARForSelfSupervisedPretraining, args: argparse.Namespace) -> None:
    """Train ``model`` with the masked-reconstruction objective and save it to ``args.output_dir``."""
    train_dataset = load_dataset(args.train_h5ad, args)
    eval_dataset = load_dataset(args.valid_h5ad, args) if args.valid_h5ad else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=5,
        load_best_model_at_end=eval_dataset is not None,
        # The input is its own target: listing it as a label makes the Trainer report eval_loss
        label_names=["cell_embedding"],
        # collate_fn needs every dataset field, not only the model inputs
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        report_to=args.report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(collate_cells, max_length=args.max_length),
    )
    trainer.train()
    trainer.save_model()


def main():
    parser = build_parser("Pre-train PULSAR from scratch")

    # Model architecture
    parser.add_argument("--input_size", type=int, default=1280, help="Dimension of the cell embeddings")
    parser.add_argument("--hidden_size", type=int, default=512, help="Size of the donor embedding")
    parser.add_argument("--encoder_num_hidden_layers", type=int, default=8, help="Transformer layers in the encoder")
    parser.add_argument("--decoder_num_hidden_layers", type=int, default=6, help="Transformer layers in the decoder")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="Attention heads in encoder and decoder")
    args = parser.parse_args()

    config = PULSARConfig(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        seq_length=args.max_length,
        encoder_num_hidden_layers=args.encoder_num_hidden_layers,
        encoder_num_attention_heads=args.num_attention_heads,
        decoder_num_hidden_layers=args.decoder_num_hidden_layers,
        decoder_num_attention_heads=args.num_attention_heads,
        use_cell_state=False,
        **masking_config(args),
    )
    train(PULSARForSelfSupervisedPretraining(config), args)


if __name__ == "__main__":
    main()
