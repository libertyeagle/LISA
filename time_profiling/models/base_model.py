import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def save_model(self, ckpt_path, epoch=None, optimizer_state_dict=None):
        if epoch is not None:
            ckpt_dict = {
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer_state_dict
            }
            torch.save(ckpt_dict, ckpt_path)
        else:
            torch.save(self.state_dict(), ckpt_path)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def restore_weights(self, ckpt_path, restore_layers_list=None):
        checkpoint = torch.load(ckpt_path)
        if type(checkpoint) == dict:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
        if restore_layers_list:
            restore_state_dict = {}
            restore_layers_set = set(restore_layers_list)
            for name, state in model_state_dict.items():
                if name.split('.')[0] in restore_layers_set:
                    restore_state_dict[name] = state
            self.load_state_dict(restore_state_dict, strict=False)
        else:
            self.load_state_dict(model_state_dict)

    def freeze_layers(self, layers_to_freeze):
        layers_to_freeze = set(layers_to_freeze)
        for name, param in self.named_parameters():
            if name.split('.')[0] in layers_to_freeze:
                param.requires_grad = False