import flwr as fl

import argparse
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')

arg_parser.add_argument('--server_round', action='store', type=int, required='True', dest='server_round')
arg_parser.add_argument('--epochs', action='store', type=int, required='True', dest='epochs')
arg_parser.add_argument('--lr', action='store', type=float, required='True', dest='lr')
arg_parser.add_argument('--batch_size', action='store', type=int, required='True', dest='batch_size')


args = arg_parser.parse_args()

id = args.id

server_round = args.server_round
epochs = args.epochs
lr = args.lr
batch_size = args.batch_size


server_address= "152.228.166.247:8081"

def get_parameters_config(server_round: int): #-> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "current_round": server_round,
            "local_epochs": epochs,
        }
    return config


def fit_config(server_round: int): #-> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "current_round": server_round,
            "local_epochs": epochs,
        }
    return config

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,
    min_fit_clients=4,
    min_available_clients=4,
    # initial_parameters = get_parameters_config, 
    on_fit_config_fn=fit_config,
    # on_evaluate_config_fn=evaluate_config
)
# print(strategy.agg)
fl.server.start_server(
    server_address=server_address,
    config=fl.server.ServerConfig(num_rounds=server_round), 
    strategy=strategy)