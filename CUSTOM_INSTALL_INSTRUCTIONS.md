## Custom Installation & Build Instructions w/ Bugfixes

1. execute ```conda env create --file=conda_env.yml```
2. execute ```cp /usr/include/crypt.h {path_to_conda_env}/envs/detectron2/include/python3.8``` *(replace ```{path_to_conda_env}```)*
3. add ```extra_link_args=['-L/usr/lib/x86_64-linux-gnu/']``` to ```ext_modules``` definition in ```detectron2/setup.py``` (after line 103)
4. execute ```python -m pip install -e detectron2```

For GripperDetection using calvin_env:
5. execute ```pip install -e projects/GripperDetection_calvin/calvin_env/tacto```
6. execute ```pip install -e projects/GripperDetection_calvin/calvin_env```
7. execute ```conda install conda-forge::lightning```
8. execute ```pip install setuptools==57.5.0```
9. execute ```python projects/GripperDetection_calvin/utils/pyhash-0.9.3/setup.py build```
10. execute ```python projects/GripperDetection_calvin/utils/pyhash-0.9.3/setup.py install```
11. execute ```pip install wandb```
12. execute ```pip install moviepy```
