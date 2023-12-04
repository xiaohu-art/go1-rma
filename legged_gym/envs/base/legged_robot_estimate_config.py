from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO


class LeggedRobotEstimateCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'OnPolicyRunnerWithEstimator'
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticWithEstimator'
        algorithm_class_name = 'PPOEstimate'
        run_name = ''
        experiment_name = 'estimate'
        max_iterations = 150000

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        estimator_loss_coef = 1.0
