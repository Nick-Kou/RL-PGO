# RL-PGO
The official repository for the paper RL-PGO: Reinforcement Learning-based Planar Pose-Graph Optimization
## Installation

The RL trainings and testing python script in the RL-PGO folder and was evaluated in a miniconda environment.
For GPU installation, steps are shown assuming training is conducted on an Nvidia RTX-3090 GPU with the supported CUDA 11.6 driver.


First refer to this [guide](https://docs.conda.io/en/latest/miniconda.html) to install conda or miniconda.

Clone the directory to your desired location:
```bash
git clone https://github.com/Nick-Kou/RL-PGO.git
```

create conda env with python 3.7:
```bash
conda create -n myenv python=3.7
```

Activate the conda environemnt:
```bash
conda activate myenv
```

install the following pip dependences:
```bash
pip install PyOpenGL
pip install numpy
pip install tensorboard
pip install pandas 
pip install matplotlib
pip install gym
```

For GPU Only - (Test on cuda 11.6 and tested on Pytorch 1.13):
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c dglteam dgl-cuda11.6
```
For CPU Only - tested on Pytorch 1.13:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c dglteam dgl
```
All C++ realted dependencies were installed using g++-7 and Tested on Ubuntu 20.04:

[How to switch between multiple GCC and G++ compiler versions on Ubuntu 20.04 LTS Focal Fossa](https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa)


Install Dependencies for Minisam - tested on Eigen 3.3.7:
```bash
sudo apt-get update
sudo apt-get install libeigen3-dev
sudo apt-get install libsuitesparse-dev
```

Install Sophus Dependency for Minisam:
```bash
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout d63ad09        (Jan 1 2020 version)
mkdir build
cd build
cmake .. 
cmake --build .
```
Install Minisam. Ensure that you activate your conda environemnt before proceeding with the next steps.
Set -DMINISAM_WITH_CUSOLVER=ON for GPU option:

```bash
conda activate yourenvname
cd /PATH/TO/RL-PGO/minisam
mkdir build
cd build
cmake .. -DMINISAM_BUILD_PYTHON_PACKAGE=ON -DMINISAM_WITH_INTERNAL_TIMING=ON -DMINISAM_WITH_SOPHUS=ON -DMINISAM_WITH_CHOLMOD=ON -DMINISAM_WITH_SPQR=ON -DMINISAM_WITH_CUSOLVER=OFF
```

Before proceeding to install minisam python package verify that the python executable is pointing to your minconda environemnt. This is found by refering to the output path after running the ```bash cmake ..``` command.

Install minisam python wrapper and package:
```bash
make 
sudo make install
sudo make python_package
sudo make install
```
Note: If you check in the myenvname/lib/python3.7/site-packages folder you may see a folder named "minisam-0.0.0-py3.7" with a folder named minisam located inside of it. If so, you must copy the minisam folder to the site-packages location and delete the folder named "minisam-0.0.0-py3.7".

For GUI (optional) refer to: [Pangolin](https://github.com/uoip/pangolin). Also ensure that you activate your conda environment such that the python wrapper is installed in the correct site packages location simular to the minisam installation process. Uncomment the "#import pangolin" line 9 in RL_PGO.py if GUI is installed.

Example command to train the proposed reccurent SAC agent:
```bash
python3 RL_PGO.py --train --no-load --frob --no-grad --no-change --no-gui --sqrt_loss --no_change_noise --lr=3e-4 --grad_value=20.0 --trans_noise=0.1495 --rot_noise=0.2 --loop_close --inter_dist=1 --prob_of_loop=0.5 --freq_of_change=1 --hidden_dim=512 --state_dim=20 --reward_numerator_scale=1.0 --reward_update_scale=10.0 --training_name=Toy_Pose_sac_v2_lstm_Training452 --num_nodes=20 --rot_action_range=0.15 --batch_size=128 --update_itr=1 --max_steps=5 --gamma=1.0 --max_episodes=230000 --max_evaluation_episodes=1
```

Example commant to test the proposed reccurent SAC agent:
```bash
python3 RL_PGO.py --test --load --frob --no-grad --no-change --no-gui --sqrt_loss --no_change_noise --lr=3e-4 --grad_value=20.0 --trans_noise=0.1495 --rot_noise=0.2 --loop_close --inter_dist=1 --prob_of_loop=0.5 --freq_of_change=1 --hidden_dim=512 --state_dim=20 --reward_numerator_scale=1.0 --reward_update_scale=10.0 --training_name=Toy_Pose_sac_v2_lstm_Training373 --num_nodes=20 --rot_action_range=0.15 --batch_size=128 --update_itr=1 --max_steps=5 --gamma=1.0 --max_episodes=230000 --max_evaluation_episodes=10
```
## Configurations Settings


Evaluate or train:
```bash
--test or --train
```
Option to load a pose-graph already placed in the run/trainingname folder or instead synthetically gernerate one during train time (graph must initially be loaded for --test mode).
```bash
--load or --no-load
```
Option to use Geodesic or Frobenius norm for rotation/orientation cost which is included in the reward function:
```bash
--no-frob or --frob
```
Option to use gradient clipping technique during training (only applicable for --train mode):
```bash
--no-grad or --grad
```
Option to use sample a graph at the specified noise parameters every "freq_of_change" episodes of training (only applicable for --train mode):
```bash
--no-change or --change
```
Option to use GUI for evaluation (only applicable for --test mode):
```bash
--no-gui or --gui
```
Option to use the square root of the sum squared orientation residuals for each factor:
```bash
--no_sqrt_loss or --sqrt_loss
```
Option to change the specified noise parameters the graphs are sampled from during training or evaluation (only applicable for --train mode):
```bash
--no_change_noise or --change_noise
```
Option to change the learning rate for training (only applicable for --train mode):
```bash
--lr=3e-4
```
Option to change the gradient clipping norm value if --grad is set for training (only applicable for --train mode):
```bash
--grad_value=20.0
```
Option to change the translational or position noise paramter in meters for sampled graph during training (only applicable for --train mode):
```bash
--trans_noise=0.1495
```
Option to change the rotataional or orientational noise paramter in radians for sampled graph during training (only applicable for --train mode):
```bash
--rot_noise=0.2
```
Option to include loop closures in the sampled graph (only applicable for --train mode):
```bash
--loop_close
```
Option to adjust the value of inter-nodal distance spacing between synthetically generated poses in meters (only applicable for --train mode):
```bash
--inter_dist=1
```
Option to change the probability a loop closure exists between factors (only applicable for --train mode). Value ranges from 0-1:
```bash
--inter_dist=1
```

Option to change the after how many episodes a new graph may be sampled (only applicable if --change is set). The command below indicates a new graph is sampled after every episode.
```bash
--freq_of_change=1
```

Option to change the hidden layer size for the recurrent LSTM unit:
```bash
--hidden_dim=512
```

Option to change the length of the encoded low dimenstional state vector output by the GNN:
```bash
--hidden_dim=512
```

Option to change numerator of the reward which is defined as numerator/(orientation cost +1):
```bash
--reward_numerator_scale=1.0
```
Option to reward scaling hyperparameter for training stabilization (only applicable for --train mode):
```bash
--reward_update_scale=10.0
```
Option to change the directory name where the training information such as the cumulative reward plot and best saved networks weights are saved:
```bash
training_name=Toy_Pose_sac_v2_lstm_Training373 
```
Option to change the number of poses in which the graph is sampled from the environment (only applicable for --train mode). Must be a factor of 10:
```bash
--num_nodes=20
```
Option to change the arbitrary rotation magnitude scale in radians that is used for SO(2) retraction by the agent on each neighborghing pose:
```bash
--rot_action_range=0.15
```

Option to change the number of episode being sampled from the replay buffer for each update during training (only applicable for --train mode):
```bash
--batch_size=128
```

Option to change the frquency of updates after the batch number of episodes have been stored in the replay buffer (only applicable for --train mode). In this case an network update occurs after every episode once 128 episodes (batch size) are completed initally:
```bash
--update_itr=1
```
Option to change the number of cycles per epsiode:
```bash
--max_steps=5
```

Option to change the gamma hyperparameter involved in the update:
```bash
--gamma=1.0 
```

Option to change the number episodes required to be completed for training to terminate:
```bash
--max_episodes=230000
```

Option to change the number of evaluation epsiodes (only applicable if --test is set):
```bash
--max_evaluation_episodes=10
```

## Citation

Please cite:
```bash
@article{Kourtzanidis2022RLPGORL,

  title={RL-PGO: Reinforcement Learning-based Planar Pose-Graph Optimization},

  author={Nikolaos Kourtzanidis and Sajad Saeedi},

  journal={ArXiv},

  year={2022},

  volume={abs/2202.13221}}
```


