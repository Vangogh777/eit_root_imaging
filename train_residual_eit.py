"""Entry point for the residual EIT route."""

import argparse

from training.residual_trainer import ResidualEITTrainer


def main():
    parser = argparse.ArgumentParser(description="Train ResidualEIT")
    parser.add_argument(
        "--config",
        default="config/residual_eit_config.yaml",
        help="ResidualEIT config path",
    )
    args = parser.parse_args()

    trainer = ResidualEITTrainer(args.config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
