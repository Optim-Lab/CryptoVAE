#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import sys
import subprocess
import re
#%%
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb
#%%
out = subprocess.run(
    ["wandb", "sweep", "sweep.yaml"], 
    capture_output=True, 
    text=True
)
cmd = out.stderr[re.search('Run sweep agent with: ', out.stderr).span()[1]:-1]
subprocess.run(cmd.split())
#%%