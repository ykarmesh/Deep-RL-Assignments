import argparse
import gym
import envs
from algo.ddpg import DDPG


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--actor_lr', dest='actor_lr', type=float,
                        default=1e-4, help="The actor's learning rate.")
    parser.add_argument('--critic_lr', dest='critic_lr', type=float,
                        default=1e-3, help="The critic's learning rate.")
    parser.add_argument('--tau', dest='tau', type=float,
                        default=0.05, help="The update parameter for soft updates")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.98, help="gamma")
    parser.add_argument('--buffer_size', dest='buffer_size', type=int, default=1000000)
    parser.add_argument('--burn_in', dest='burn_in', type=int, default=5000)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1024)
    parser.add_argument('--epsilon', dest='epsilon', type=float,
                        default=0.1, help='Epsilon for noise')
    #TD3
    parser.add_argument('--target_action_sigma', dest='target_action_sigma', type=float,
                        default=0.05, help='action smoothing for TD3.')
    parser.add_argument('--clip', dest='clip', type=float,
                        default=0.05, help='clip value for action smoothing in TD3.')
    parser.add_argument('--policy_update_frequency', dest='policy_update_frequency', type=int,
                        default=2, help='How fast should we update policy wrt critic')
    parser.add_argument('--num_update_iters', dest='num_update_iters', type=int,
                        default=4, help='How many times to update the critic every episode')

    # parser.add_argument('--test_episodes', dest='test_episodes', type=int,
    #                     default=100, help='Number of episodes to test` on.')
    parser.add_argument('--save_interval', dest='save_interval', type=int,
                        default=2000, help='Weights save interval.')
    parser.add_argument('--test_interval', dest='test_interval', type=int,
                        default=100, help='Test interval.')
    parser.add_argument('--log_interval', dest='log_interval', type=int,
                        default=100, help='Log interval.')
    parser.add_argument('--weights_path', dest='weights_path', type=str,
                        default=None, help='Pretrained weights file.')
    parser.add_argument('--train', action="store_true", default=True,                    
                        help='Do training')
    parser.add_argument('--algorithm', type=str, default='ddpg',
                        help='Training algorithm (ddpg | td3 | her)')
    parser.add_argument('--custom_init', action="store_true", default=False,                    
                        help='If the netowrk should be initialized using custom weight values')
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()

def main():
    args = parse_arguments()
    env = gym.make('Pushing2D-v0')
    algo = DDPG(args, env, 'ddpg_log.txt')
    algo.train(args.num_episodes)


if __name__ == '__main__':
    main()
