# OpenTraj Toolkit

The official implementation for the paper:

**OpenTraj: Assessing Prediction Complexity in Human Trajectories Datasets**  
*[Javad Amirian](http://people.rennes.inria.fr/Javad.Amirian),
[Bingqing Zhang](),
[Francisco Valente Castro](),
[Juan José Baldelomar](),
[Jean-Bernard Hayet](http://aplicaciones.cimat.mx/Personal/jbhayet/home),
[Julien Pettré](http://people.rennes.inria.fr/Julien.Pettre/)*  
Published at [ACCV2020](http://accv2020.kyoto/): [[paper](https://arxiv.org/abs/2010.00890)], [[presentation]()]



## Dataset Analysis

We present indicators in 3 categories:
- Predictability
    - Conditional Entropy [`conditional_entropy.py`](indicators/trajectory_entropy.py)
    - Number of Clusters [`global_multimodality.py`](indicators/global_multimodality.py)
        - The algorithm implemented here reparametrizes all the trajectories in the dataset using a cubic spline. We can take samples of points at specific times (like t = 1.0) for all the trajectories and then calculate the number of clusters and adjust a Gaussian Mixture Model. The following image describes the method for t = 1.0 on the ETH dataset.  <p align="center"> 
            <img src="../../docs/figs/fig-opentraj-eth-global-multimodality.png" alt="global multimodality" width="350" />
        </p>
    
- Trajlet Regularity
    - Average Speed 
    - Speed Range
    - Average Acceleration
    - Maximum Acceleration
    - Path Efficiency
    - Angular Deviation
    
- Context Complexity
    - Distance to Closest Approach (DCA)
    - Time-to-Collision (TTC)
    - Local Density


## Setup
To set up the code and to run the benchmarking, run the following script:

```bash
# create a virtualenv and activate it
python3 -m venv env
source env/bin/activate

# install dependencies
cd [OpenTraj]
pip install -r benchmarking/requirements.txt

# run it!
python toolkit/benchmarking . [output_dir]

# exit the environment
deactivate  # Exit virtual environment
``` 


