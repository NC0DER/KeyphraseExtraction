import os
import pandas as pd
import KeyExt.metrics
import KeyExt.utils

def run_experiments(datasets_dir, output_dir, top_n = 10, partial_match = True):

    # Make a list of all subdirectories.
    directories = next(os.walk(datasets_dir))[1][0:]
    data = []

    # Set the metric name and construct the output path for the xlsx.
    metric_name = f'pF1@{top_n}' if partial_match else f'F1@{top_n}'
    xlsx_path = os.path.join(output_dir, f'{metric_name}.xlsx')
    print(f'Calculating the {metric_name} score for all datasets...')

    for i, directory in enumerate(directories):
        print(f'Processing {i+1} in {len(directories)} datasets.')

        # Change current working directory to the dataset directory.
        dataset_path = os.path.join(datasets_dir, directory)
        os.chdir(dataset_path)

        # Find human assigned keyphrase files and paths.
        os.chdir(os.path.join(dataset_path, 'keys'))
        key_paths = list(map(os.path.abspath, sorted(os.listdir())))

        # Find all methods (directories of keys) and their generated keyphrase files and paths.
        extracted_path = os.path.join(dataset_path, 'extracted')
        os.chdir(extracted_path)
        methods = sorted(next(os.walk('.'))[1])

        # Initialize the macro(mean) metric vector.
        macro_metric_vec = [0.0] * len(methods)

        # Compare the extracted keys of each method with the human assigned keys.
        for j, method in enumerate(methods):

            print(f'    * Evaluating {method} for {len(key_paths)} documents.')

            # Find all extracted keys of the method.
            os.chdir(os.path.join(extracted_path, method))
            method_paths = list(map(os.path.abspath, sorted(os.listdir())))

            for key_path, method_path in zip(key_paths, method_paths):
                with open(method_path, 'r', encoding = 'utf-8-sig', errors = 'ignore') as method_keys, \
                     open(key_path, 'r', encoding = 'utf-8-sig', errors = 'ignore') as human_keys:
                    
                    # Read the tags from file and then preprocess them, 
                    # as to be lowercased, with no punctuation and stemmed.
                    extracted = KeyExt.utils.preprocess(method_keys.read().split('\n'))
                    assigned = KeyExt.utils.preprocess(human_keys.read().split('\n'))
                    macro_metric_vec[j] += KeyExt.metrics.f1_metric_k (
                    assigned, extracted, k = top_n, partial_match = partial_match
                )

        # The macro (mean) metric score us calculated from each method.
        macro_metric_vec = [
            round(metric_sum / len(key_paths), 3)
            for metric_sum in macro_metric_vec
        ]

        # Append the macro metric score for each directory to the data list of lists,
        # each list has the dataset name prepended at the start of the row.
        data.append([directory] + macro_metric_vec)
        os.system('clear')


    # Construct the dataframe and then transpose it.
    df = pd.DataFrame(data, columns = [f'{metric_name}', *methods]).set_index(f'{metric_name}')
    df = df.transpose()

    # Save the dataframe to excel.
    df.to_excel(xlsx_path, engine = 'openpyxl')
    return
