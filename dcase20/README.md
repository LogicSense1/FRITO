# Experiments on DCASE20

```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/ba3l/lib python -m debugpy --listen 0.0.0.0:5678 ex_dcase20_dev.py with models.net.s_patchout_t=10 models.net.s_patchout_f=5 trainer.use_tensorboard_logger=True -p --debug
```

