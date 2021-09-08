from mmf.utils.modeling import *

# =================================================
#              Optimizer Parameters
# =================================================  

def get_optimizer_parameters_custom(config, module):
    """
        A function that trains at different learning rates for
        lower layers vs the classifier the via finetune_lr_multiplier.
        Copied from https://github.com/facebookresearch/mmf/blob/133b923938ae785de3baf711624b54813849ef4e/mmf/utils/modeling.py#L49.
    """
    lr = config.optimizer.params.lr
    model_config = getattr(config.model_config, config.model, {})
    finetune_lr_multiplier = getattr(model_config, "finetune_lr_multiplier", 1)

    # For pretraining or when finetune_lr_multiplier == 1, all modules will be trained
    # with default lr.
    if module.config.training_head_type == "pretraining" or finetune_lr_multiplier == 1:
        return get_bert_configured_parameters(module)

    # For non pretraining heads, where finetune_lr_multiplier != 1, all modules other
    # than classifier will be trained with (lr * finetune_lr_multiplier).
    has_soft_attention = False
    parameters = []
    for name, submodule in module.named_children():
        if name == "classifier":
            continue
        parameters += get_bert_configured_parameters(
            submodule, lr * finetune_lr_multiplier
        )
        print(f"Overriding {name} module's LR to {lr * finetune_lr_multiplier}")
    # Classifier will be trained with default lr.
    parameters += get_bert_configured_parameters(module.classifier)
    return parameters

def freeze_optimizer_parameters_clip(config, module):
    """
        A function that freezes finetune_layers in CLIP based on 
        config arguments like freeze_lower and freeze_all.
    """
    if config.get("freeze_lower", False) or config.get("freeze_all", False):
        finetune_layers = ["classifier", "visual.layer4", "visual.attnpool", "transformer.resblocks.11", "ln_final", "text_projection", "logit_scale"]
        if config.get("freeze_all", False):
            finetune_layers = []

        print(f"Freezing CLIP's parameters: {finetune_layers}")

        detected = []
        for name, submodule in module.named_parameters():
            flag = True
            for layer_type in finetune_layers:
                if layer_type in name:
                    detected.append(layer_type)
                    flag = False
            if flag:
                submodule.requires_grad = False
        print("Layers not detected: ", set(finetune_layers) - set(detected))
    else:
        print("NOT freezing CLIP's lower parameters")
