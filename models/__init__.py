from .bert import BERTModel
from .dae import DAEModel
from .vae import VAEModel
from .bert_ls import BERTLSModel
MODELS = {
    BERTModel.code(): BERTModel,
    BERTLSModel.code(): BERTLSModel,
    DAEModel.code(): DAEModel,
    VAEModel.code(): VAEModel
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
