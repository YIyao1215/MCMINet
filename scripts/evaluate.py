"""Explicit evaluation entrypoint; no threshold optimization or external-data discovery."""
import argparse

def main():
    p=argparse.ArgumentParser(description='Evaluate an explicitly supplied MCMINet checkpoint and manifest.')
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    args=p.parse_args()
    from mcminet.config import load_config
    load_config(args.config)
    raise SystemExit('Evaluation requires local prepared data; no automatic threshold or cohort selection is performed.')
if __name__=='__main__':main()
