import os
import pandas as pd
from pathlib import Path


def create_temp_dirs(tmp_dir):
    Path(os.path.join(tmp_dir, "models")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(tmp_dir, "metrics")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(tmp_dir, "projection")).mkdir(parents=True, exist_ok=True)


class log_obj:
    """Writes the run's parameters and one metrics file per tracked value."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.fhands = {}
    
    def log_params(self,args):
        dfparams = pd.DataFrame(data=vars(args),index=['value']).transpose()
        dfparams.to_csv(os.path.join(self.run_dir, 'models', 'params.csv'))
        
    def log_metric(self,name,value,epoch):
        if name not in self.fhands.keys():
            fhand = open(os.path.join(self.run_dir,'metrics',name),'w',buffering=1)
            fhand.write('epoch\tvalue\n')
            self.fhands[name] = fhand
            
        self.fhands[name].write(f'{epoch}\t{value}\n')
    
    def end_log(self):
        for fhand in self.fhands.values(): fhand.close()
