import argparse

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument('--train_batch', type=int, default=64)
    parser.add_argument('--test_batch', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)

    # compression framework 
    parser.add_argument('--e_dim', type=int, default=16, help='Dimension of each codeword') 
    parser.add_argument('--n_e', type=int, default=4, help='Codebook size') 

    # show bits
    parser.add_argument('--show_bits', type=bool, default=False, help='Show bitstream length of each image')

    # training setting
    parser.add_argument('--load_pretrained', type=bool, default=True)
    parser.add_argument('--epoch', default = 100)
    parser.add_argument('--lr', default = 1e-5)
    parser.add_argument('--lr_step_size', default = 20)
    parser.add_argument('--lr_gamma', default = 0.1)
    parser.add_argument('--weight_decay', default = 5e-04)
    parser.add_argument('--alpha', default = 10)
    parser.add_argument('--beta', default = 0.25)
    parser.add_argument('--optimizer', default = "Adam")
    parser.add_argument('--num_samples_per_class', default = 50)

    # caption (inherited from https://github.com/rmokady/CLIP_prefix_caption)
    parser.add_argument('--prefix_size', default=768)
    parser.add_argument('--prefix_length', default=10) 
    parser.add_argument('--save_every', default=1)

    args = parser.parse_args()

    return args