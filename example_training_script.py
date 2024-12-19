import argparse
from framework.cpu_trainer import CPUTrainer
from framework.gpu_trainer import GPUTrainer

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model using CPU or GPU.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu',
                        help="Specify the device for training: 'cpu' or 'gpu'. Default is 'cpu'.")
    args = parser.parse_args()

    # Configuration for CPU training
    cpu_config = {
        'model_config': {
            'epochs': 5,
            'batch_size': 32,
            'save_path': '/models/cpu_model.h5'
        },
        'data_config': {
            'data_path': '/data'
        }
    }
    
    # Configuration for GPU training
    gpu_config = {
        'model_config': {
            'epochs': 10,
            'batch_size': 64,
            'save_path': '/models/gpu_model.h5'
        },
        'data_config': {
            'data_path': '/data'
        }
    }

    if args.device == 'cpu':
        print("Training on CPU...")
        trainer = CPUTrainer(cpu_config['model_config'], cpu_config['data_config'])
        history = trainer.train()
    elif args.device == 'gpu':
        print("Training on GPU...")
        trainer = GPUTrainer(gpu_config['model_config'], gpu_config['data_config'])
        history = trainer.train()

if __name__ == "__main__":
    main()
