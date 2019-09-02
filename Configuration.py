class Config:

    env = str(None)
    episodes = int(0)
    max_steps = 1000
    max_buff_size = int(0)
    batch_size = int(0)
    state_num = int(0)
    state_high = None
    state_low = None
    seed = None

    output = 'out'

    action_num = int(0)
    action_high = None
    action_low = None
    action_lim = None

    learning_rate_actor = float(0.0)
    learning_rate_critic = float(0.0)

    gamma = float(0.0)
    tau = float(0.0)
    epsilon = float(0.0)
    eps_decay = None
    epsilon_min = float(0.0)

    RENDER = True
    
    # use_cuda: bool = True

    # checkpoint: bool = False
    # checkpoint_interval: int = None

    # use_matplotlib: bool = False

    # record: bool = False
    # record_ep_interval: int = None
