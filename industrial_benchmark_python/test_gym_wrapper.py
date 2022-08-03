from industrial_benchmark_python.IBGym import IBGym
import numpy as np

from gym.utils.env_checker import check_env

DISCOUNT = 0.97

env = IBGym(70, new_step_api=False, action_type="continuous")
check_env(env)  # environment must confirm gym-api
env.reset()
returns = []
for _ in range(100):
    acc_return = 0.0
    for i in range(100):
        state, reward, done, info = env.step(env.action_space.sample())
        acc_return += reward * DISCOUNT**i
    returns.append(acc_return / 100.0)

print("random actions achieved return", np.mean(returns), "+-", np.std(returns))
