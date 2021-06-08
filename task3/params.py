import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--python_seed", type=int, default=49)
parser.add_argument("--torch_seed", type=int, default=4214)

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--n_epochs", type=int, default=5)

parser.add_argument("--me", dest="me", action="store_true")
parser.add_argument("--mp", dest="mp", action="store_true")
parser.add_argument("--qual", dest="qual", action="store_true")
parser.add_argument("--token_type_ids_not", dest="token_type_ids_not", action="store_true")

parser.add_argument("--test_label_folder", type=str, required=True)

parser.add_argument("--bert_type", type=str, required=True)
parser.add_argument("--dummy_run", dest="dummy_run", action="store_true")
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")

params = parser.parse_args()

assert params.test_label_folder != "NONE"
assert params.me or params.mp or params.qual
print(params)
