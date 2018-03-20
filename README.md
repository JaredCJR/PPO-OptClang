PPO-OptClang
===============================================
Implementation of proximal policy optimization(PPO) using tensorflow with custom Gym Environment(OptClang-gym)

Environment
----------------------------------------
Refer to [gym-OptClang](https://github.com/JaredCJR/gym-OptClang)

Dependencies
----------------------------------------
* Python 3.5
* Tensorflow 1.5
* OpenAI Gym 0.9.4

Training
----------------------------------------
`python3 run.py`  
`python3 ./run.py -h` for more details.  
`config.json` contains the hyper-parameters for RL-model and workers.  

Inference
---------------------------------------
Refer to [ThesisTools](https://github.com/JaredCJR/ThesisTools#how-to-use-the-trained-model-to-inference)

Test trained policy
----------------------------------------
`python3 test_policy.py`

Tensorboard
----------------------------------------
`tensorboard --logdir log`

License
----------------------------------------
Refer to [LICENSE](./LICENSE)
