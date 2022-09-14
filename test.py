import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'positional_argument',
        help='this is a positional argument',
    )
    parser.add_argument(
        '--arg1',
        help='the first argument',
        type=int,
        dest='arg1',
        default=1,
    )
    parser.add_argument(
        '--arg2',
        help='the second argument',
        type=str,
        dest='arg2',
    )

    args = parser.parse_args()
    pos_argument = args.positional_argument
    arg1 = args.arg1
    arg2 = args.arg2