import argparse
from trainer import train_model
def main(args):
    config = {
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'resize_dim': args.resize_dim,
        'upscale_factor': args.upscale_factor,
        'batch_size': args.batch_size,
        'device': args.device,
        'res_block':args.res_block,
        'train_path':args.train_path,
        'val_path':args.val_path,
        'mode':args.mode
    }
    train_model(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--res_block', type=int, default=1)
    parser.add_argument('--resize_dim', type=int, default=512)
    parser.add_argument('--upscale_factor', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--mode', type=str, required=False, default='train')
    parser.add_argument('--train_path', type=str, required=False, default='./archive/DIV2K_train_HRss/DIV2K_train_HR')
    parser.add_argument('--val_path', type=str, required=False, default='./archive/DIV2K_valid_HR/DIV2K_valid_HR')
    arguments = parser.parse_args()
    main(arguments)
