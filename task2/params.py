import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--quant_path", type=str, required=True)

params = parser.parse_args()
