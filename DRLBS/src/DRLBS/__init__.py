from gym.envs.registration import register

register(id='BS-v0',
         entry_point='DRLBS.envs:BandSelection',
         nondeterministic=True)

register(id='BSEnv-v0',
         entry_point='DRLBS.envs:BSEnvironment',
         nondeterministic=True)