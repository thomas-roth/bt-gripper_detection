## Custom Installation & Build Instructions w/ Bugfixes

1. execute ```conda env create --file=conda_env.yml```
2. execute ```cp /usr/include/crypt.h {path_to_conda_env}/envs/detectron2/include/python3.8``` *(replace ```{path_to_conda_env}```)*
3. add ```extra_link_args=['-L/usr/lib/x86_64-linux-gnu/']``` to ```ext_modules``` definition in ```detectron2/setup.py``` (after line 103)
4. execute ```python -m pip install -e detectron2```
