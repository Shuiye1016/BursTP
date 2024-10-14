# BursTP
A Deep Burst Time Prediction Framework for Information Diffusion in Social Networks

## Description
Our BursTP is implemented mainly based on the following libraries (see the README file in source code folder for more details):

- PyTorch: https://pytorch.org/
- PyTorch Lightning: https://www.pytorchlightning.ai/
- DGL: https://www.dgl.ai/
- PytorchNLP: https://github.com/PetrochukM/PyTorch-NLP
- Networkx: https://networkx.org/

### File Tree

Our main project file structure and description:

```python
BursTP
├─ README.md
├─ dataset    # All data will be made publicly available upon receipt of the paper
│    ├─ twitter       # Our twitter dataset
├─ model       # package of models (BursTP)
│    ├─ BursTP.py
│    ├─ Burstformer.py
│    ├─ embedding.py
│    ├─ MyDataset.py
│    ├─ utils.py
├─ process       # package of preprocessing
│    ├─ excute_burst.py
│    ├─ process_burst.py
│    ├─ gen_globalg.py
│    ├─ key_sample.py
│    ├─ split.py
│    ├─ data_transforms.py
│    ├─ global_params.py
│    ├─ params.py
│    ├─ utils.py
└─ train.py	      # You can train directly
```

### Installation

Installation requirements are described in `environment.txt`

- Create Conda Environment:

  Open a terminal or command prompt and run the following command:

  ```bash
  conda env create -f environment.yml
  ```

- Activate Conda Environment:

  After the installation is complete, activate the newly created Conda environment using the following command:

  ```bash
  conda activate <environment_name>
  ```
  
- Verify Environment Installation:

  You can verify whether the environment is correctly installed:

  ```bash
  conda list
  ```

## Usage

You can directly start training.

```bash
python train.py
```

### Experiment Settings

Please refer to sections 5.1.1 and 5.1.3 of the paper for details on data and parameter settings.
