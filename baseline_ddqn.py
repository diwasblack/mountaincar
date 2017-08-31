import gym

from baselines import deepq


def main():
    env = gym.make("MountainCar-v0")
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([512, 256], layer_norm=True)
    act = deepq.learn(
        env,
        q_func=model,
        lr=0.001,
        max_timesteps=200000,
        buffer_size=2000,
        exploration_fraction=0.05,
        exploration_final_eps=0.01,
        print_freq=10,
        batch_size=32,
        param_noise=True
    )
    print("Saving model to mountaincar_model.pkl")
    act.save("mountaincar_model.pkl")


if __name__ == '__main__':
    main()
