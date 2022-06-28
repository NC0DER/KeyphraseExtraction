from KeyExt.config import datasets_path, output_dir
from KeyExt.experiments import run_experiments


def main():
    for partial_match in [False, True]:
        for n in [5, 10]:
            run_experiments(
                datasets_path, output_dir, 
                top_n = n, partial_match = partial_match
            )


if __name__=='__main__': main()
