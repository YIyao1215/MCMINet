"""Explicit training entrypoint; raw data paths must be supplied by the caller."""
import argparse

def main():
    p=argparse.ArgumentParser(description='Train MCMINet with an explicit manifest and internal validation split.')
    p.add_argument('--config',required=True);p.add_argument('--manifest');p.add_argument('--output',required=True)
    p.add_argument('--validate-config', action='store_true', help='Validate and print the candidate protocol without accessing cohorts')
    args=p.parse_args()
    from mcminet.config import load_config, require_candidate
    config = load_config(args.config)
    require_candidate(config)
    if args.validate_config:
        import json
        print(json.dumps(config, indent=2))
        return
    raise SystemExit('Training requires caller-supplied prepared cohorts and a validated cache; the packaged example contains no medical data.')
if __name__=='__main__':main()
