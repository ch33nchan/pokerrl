import argparse
import torch
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from qagent.agents.dcfr_leduc_lstm import LSTMAgentNet, CARD_EMBEDDING_DIM
from qagent.agents.dcfr_leduc import RegretNet as AdvantageNetwork
from qagent.agents.dcfr_leduc_transformer import TransformerStrategyNet as Transformer_DMC

def count_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description="Calculate the number of parameters for a given model architecture.")
    parser.add_argument('architecture', type=str, choices=['lstm', 'baseline', 'transformer'], help='The model architecture to calculate parameters for.')

    # Arguments for Baseline
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 128], help='Hidden layer sizes for the baseline model.')

    # Arguments for LSTM
    parser.add_argument('--lstm_hidden_size', type=int, default=128, help='Hidden size for the LSTM layers.')
    parser.add_argument('--num_lstm_layers', type=int, default=2, help='Number of LSTM layers.')
    parser.add_argument('--lstm_dropout', type=float, default=0.0, help='Dropout rate for LSTM layers.')

    # Arguments for Transformer
    parser.add_argument('--transformer_d_model', type=int, default=128, help='Dimension of the model for the Transformer.')
    parser.add_argument('--transformer_nhead', type=int, default=4, help='Number of heads in the multiheadattention models.')
    parser.add_argument('--transformer_num_layers', type=int, default=2, help='Number of sub-encoder-layers in the encoder.')
    parser.add_argument('--transformer_dim_feedforward', type=int, default=256, help='Dimension of the feedforward network model.')
    parser.add_argument('--transformer_dropout', type=float, default=0.1, help='Dropout value.')


    args = parser.parse_args()

    # Common parameters for all models
    # These values are specific to Leduc Hold'em and need to be accurate
    # for model instantiation.
    state_dim_for_lstm_transformer = 38
    state_dim_baseline = 10 # From the baseline agent file
    action_dim = 3

    model = None
    if args.architecture == 'baseline':
        # The AdvantageNetwork is the model used within the baseline DCFR agent
        model = AdvantageNetwork(hidden_sizes=args.hidden_sizes)
        print(f"Calculating parameters for Baseline model with hidden_sizes={args.hidden_sizes}")

    elif args.architecture == 'lstm':
        model = LSTMAgentNet(
            embedding_dim=CARD_EMBEDDING_DIM, # Using the constant from the agent file
            lstm_hidden_size=args.lstm_hidden_size,
            num_lstm_layers=args.num_lstm_layers,
        )
        print(f"Calculating parameters for LSTM model with hidden_size={args.lstm_hidden_size}, num_layers={args.num_lstm_layers}")

    elif args.architecture == 'transformer':
        model = Transformer_DMC(
            state_dim=state_dim_for_lstm_transformer,
            action_dim=action_dim,
            d_model=args.transformer_d_model,
            nhead=args.transformer_nhead,
            num_encoder_layers=args.transformer_num_layers,
            dim_feedforward=args.transformer_dim_feedforward,
            dropout=args.transformer_dropout
        )
        print(f"Calculating parameters for Transformer model with d_model={args.transformer_d_model}, nhead={args.transformer_nhead}, num_layers={args.transformer_num_layers}")
    if model:
        num_params = count_parameters(model)
        print(f"Total trainable parameters: {num_params:,}")
    else:
        print("Could not create the model. Please check the architecture name.")

if __name__ == '__main__':
    main()
