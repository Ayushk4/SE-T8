import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--python_seed", type=int, default=49)
parser.add_argument("--torch_seed", type=int, default=4214)
parser.add_argument("--test_mode", dest="test_mode", action="store_true", help="")

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--n_epochs", type=int, default=5)

parser.add_argument("--task", type=str, default="QUANT")
parser.add_argument("--bert_type", type=str, required=True)
parser.add_argument("--dummy_run", dest="dummy_run", action="store_true")
parser.add_argument("--from_tf", dest="from_tf", action="store_true")
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")

parser.add_argument("--run", type=str, default=None)

params = parser.parse_args()

assert params.task in ["QUANT", "ME", "MP", "QUAL"]
