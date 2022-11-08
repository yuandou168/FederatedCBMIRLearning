import flwr as fl

server_address= "152.228.166.247:8081"
# server_round = 3
# epochs = 10 

# server_round = 1
# epochs = 30

# server_round = 2
# epochs = 15

server_round = 2
epochs = 20

def get_parameters_config(server_round: int): #-> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
            # "learning_rate": 0.001,
            # "batch_size": batch_size,
            "current_round": server_round,
            "local_epochs": epochs,
        }
    return config


def fit_config(server_round: int): #-> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
            # "learning_rate": 0.001,
            # "batch_size": batch_size,
            "current_round": server_round,
            "local_epochs": epochs,
        }
    return config

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,
    min_fit_clients=2,
    min_available_clients=2,
    # initial_parameters = get_parameters_config, 
    on_fit_config_fn=fit_config,

    # on_evaluate_config_fn=evaluate_config
)
fl.server.start_server(
    server_address=server_address,
    config=fl.server.ServerConfig(num_rounds=server_round), 
    strategy=strategy)
