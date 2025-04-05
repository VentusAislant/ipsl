from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class IpslFreezeLayersHook(Hook):


    def __init__(self):
        super().__init__()

    def before_train(self, runner) -> None:
        # for name, param in runner.model.named_parameters():
        #     param.requires_grad_(False)

        for name, param in runner.model.named_parameters():
            if 'Ipsl_module.vision_tower' in name:
                param.requires_grad_(False)
            if 'Ipsl_module.semantic_projector' in name:
                param.requires_grad_(False)

        print('=' * 90)
        print(runner.model)
        print('=' * 90)
        print('Trainable params: ')
        for n, p in runner.model.named_parameters():
            if p.requires_grad:
                print(n)
        print('=' * 90)
