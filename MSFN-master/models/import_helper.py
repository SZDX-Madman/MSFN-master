from utils import load_module


def get_model(model_path):
    mod_model = load_module(model_path)
    return mod_model.get()

