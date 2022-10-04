from .bert import BERTTrainer
from .dae import DAETrainer
from .vae import VAETrainer
from .bertls import BERTLSTrainer

TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    BERTLSTrainer.code(): BERTLSTrainer,
    DAETrainer.code(): DAETrainer,
    VAETrainer.code(): VAETrainer
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
