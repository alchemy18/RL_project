from DRLBS.envs.BS import BandSelection

class BSEnvironment(BandSelection):
    def __init__(self, 
                n_bands: int = 200,  
                max_bands: int = 30,
                reward_penalty: float = 0.01,
                accuracy_threshold: float = 0.70,
                weights_path: str = 'D:\\IIITD\\RL\\Project\\DRLBS\\src\\DRLBS\\models\\best_model_baseline.pth',
                data_path: str = 'D:\\IIITD\\RL\\Project\\DRLBS\\src\\DRLBS\\models\\indian_pines_randomSampling_0.1_run_1.mat',
                batch_size: int = 64
    ):
        state_observables = [0]*n_bands
        actions = [i for i in range(0, n_bands)]
        super(BSEnvironment,self).__init__(n_bands, max_bands, state_observables, actions, reward_penalty, accuracy_threshold, weights_path, data_path, batch_size)