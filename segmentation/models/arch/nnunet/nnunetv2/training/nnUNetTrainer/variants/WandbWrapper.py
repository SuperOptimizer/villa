import wandb
from datetime import datetime

class WandbWrapper():
    def __init__(self, use_wandb=True, config=None):
        self.use_wandb = use_wandb

        if self.use_wandb:
            self.config = config

    def init(self, local_rank=0):
        if self.use_wandb:
            if local_rank == 0:
                nnow = datetime.now()
                ndate_str = nnow.strftime("%Y.%m.%d")
                ntime_str = nnow.strftime("%H:%M:%S")
                wandb_name = f"{self.config['project_name']}_{self.config['architecture']}_fold_{self.config['fold']}_rank{local_rank}_{ndate_str}_{ntime_str}"
                return wandb.init(project=self.config['project_name'], name=wandb_name, config=self.config)
            else:
                # disable wandb logging for non-main processes
                return wandb.init(mode="disabled")
    
    def log(self, *args, **kwargs):
        if self.use_wandb:
            return wandb.log(*args, **kwargs)

    def finish(self):
        if self.use_wandb:
            return wandb.finish()
