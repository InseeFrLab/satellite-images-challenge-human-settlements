import sys
from pipeline.pipeline_launcher import run_pipeline


if __name__ == "__main__":
    # MLFlow param
    run_name = sys.argv[1]
    run_pipeline(run_name)
