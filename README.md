# FlyProjection

<!-- badges: start -->
<!-- badges: end -->

Author: [Rishika Mohanta](https://neurorishika.github.io/)

Latest Build Date: 2024-07-23 16:37:35

## About the Project

Project description is being updated. Please check back later.

## Instructions

This is a [Poetry](https://python-poetry.org/)-enabled python project. Poetry installs a virtual environment in the project directory and all packages are installed in this virtual environment. This means that you do not need to install any packages in your system. The virtual environment is automatically activated when you run the project through Poetry. 

If you use [VS Code](https://code.visualstudio.com/), you can set the Python interpreter to the Poetry virtual environment `.venv` in the project directory for script execution and debugging and use the Poetry virtual environment `.venv` for the Jupyter kernel.

First, you need to setup a git alias for tree generation by running the following command on the terminal:

```
git config --global alias.tree '! git ls-tree --full-name --name-only -t -r HEAD | sed -e "s/[^-][^\/]*\//   |/g" -e "s/|\([^ ]\)/|-- \1/"'
```

To run the project, make sure you have Poetry installed and run the following commands in the project directory:

```
poetry run python utils/update.py
poetry run python utils/build.py
```

To run the Jupyter notebook, run the following command in the project directory:

```
poetry run jupyter notebook
```

## Project Organization

The project is organized as follows:
```
.DS_Store
.gitignore
LICENSE
README.md
analysis
   |-- .gitkeep
   |-- 20hr-wingless-orco-yy
   |   |-- analysis.ipynb
   |   |-- arena.json
   |   |-- video_gen.ipynb
   |   |-- yang_props.json
   |   |-- ying_props.json
   |-- 20hrs-wingless-orcoctrl-yy2024-04-26_13-24
   |   |-- analysis.ipynb
   |   |-- arena.json
   |   |-- video_gen.ipynb
   |   |-- yang_props.json
   |   |-- ying_props.json
   |-- OLD METHOD
   |   |-- analysis.ipynb
   |-- Thin-Trails_ORCO
   |   |-- 20hr-wingless-orco-tt
   |   |   |-- analysis.ipynb
   |   |   |-- arena.json
   |   |   |-- big_ring_props.json
   |   |   |-- small_ring_props.json
   |   |-- 20hr-wingless-orcoctrl-tt
   |   |   |-- analysis-archived-2.ipynb
   |   |   |-- analysis-archived.ipynb
   |   |   |-- analysis.ipynb
   |   |   |-- arena.json
   |   |   |-- big_ring_props.json
   |   |   |-- small_ring_props.json
   |-- archived
   |   |-- 20hr-wingless-orcoctrl-yy-BADTRACKING
   |   |   |-- analysis.ipynb
   |   |   |-- arena.json
   |   |   |-- video_gen.ipynb
   |   |   |-- yang_props.json
   |   |   |-- ying_props.json
   |-- process_bands.ipynb
   |-- process_thin_trails.ipynb
   |-- process_ying-yang-oc.ipynb
   |-- simulation
   |   |-- analysis.ipynb
   |   |-- simulation.ipynb
   |   |-- test.ipynb
configs
   |-- archived_configs
   |   |-- rig_config_20240416120834.json
   |   |-- rig_config_20240419111528.json
   |-- rig_config.json
data
   |-- .gitkeep
experiments
   |-- dual_band
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- dual_trail
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- multi_trail_ece
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- multi_trail_mov
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- ortho_circle
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- patches
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- random_flash
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- trail_test
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- yin_yang
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- yy_oc
   |   |-- config.py
   |   |-- experiment_logic.py
flyprojection
   |-- __init__.py
   |-- config.py
   |-- controllers
   |   |-- __init__.py
   |   |-- camera.py
   |-- experiment_logic.py
   |-- main.py
   |-- rdp_client.py
   |-- reanalysis.py
   |-- rig-reconfig.py
   |-- utils.py
   |-- webapp.py
poetry.lock
poetry.toml
processed_data
   |-- .gitkeep
project_readme.md
push_script.sh
pyproject.toml
scripts
   |-- .gitkeep
tests
   |-- __init__.py
utils
   |-- build.py
   |-- quickstart.py
   |-- update.py
```
