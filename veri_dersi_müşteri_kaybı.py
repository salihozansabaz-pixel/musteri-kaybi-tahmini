"""Backward-compatible entry point; the implementation lives in churn.py."""
from churn import build_models, main, make_splits, prepare_data

if __name__ == '__main__':
    main()
