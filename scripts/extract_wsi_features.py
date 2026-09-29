import argparse
p=argparse.ArgumentParser(description='extract_wsi_features for caller-supplied local data')
p.add_argument('--input');p.add_argument('--output')
if __name__=='__main__': p.parse_args()
