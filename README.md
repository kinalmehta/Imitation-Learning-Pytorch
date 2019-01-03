# Imitation Learning PyTorch

Dependencies:
 * Python **3.6**
 * Numpy version **1.14.5**
 * TensorFlow
 * Pytorch version **1.0**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym
 * tqdm

The pretrained models trained with dagger and expert models have been provided. Preferably use pretrained models for first viewing how the models perform.
The expert models are from "https://github.com/berkeleydeeprlcourse/homework/tree/master/hw1"
Run the code as follows:
```
python run.py --envname envName --num_rollouts 50 --max_timesteps 1000 --render --use_pretrained --cloning
python run.py --envname envName --num_rollouts 50 --max_timesteps 1000 --render --use_pretrained --dagger
                                                    (optional)         (optional)   (optional)
```

**Note**: Though the option for behaviour cloning is given, the models trained with dagger perform way better. The results when only behaviour cloning is used are below par. And hence only pre-models trained with dagger are provided. 


The available environments are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl
