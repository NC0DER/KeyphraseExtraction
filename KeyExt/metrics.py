def exact_f1_k(assigned, extracted, k):
    """
    Computes the exatch match f1 measure at k.
    Arguments
    ---------
    assigned  : A list of human assigned keyphrases.
    extracted : A list of extracted keyphrases.
    k         : int
                The maximum number of extracted keyphrases.
    Returned value
    --------------
              : double
    """
    # Exit early, if one of the lists or both are empty.
    if not assigned or not extracted:
        return 0.0

    precision_k = len(set(assigned) & set(extracted)) / k
    recall_k = len(set(assigned) & set(extracted)) / len(assigned)
    return (
        2 * precision_k * recall_k / (precision_k + recall_k)
        if precision_k and recall_k else 0.0
    )


def partial_f1_k(assigned, extracted, k):
    """
    Computes the exatch match f1 measure at k.
    Arguments
    ---------
    assigned  : A list of human assigned keyphrases.
    extracted : A list of extracted keyphrases.
    k         : int
                The maximum number of extracted keyphrases.
    Returned value
    --------------
              : double
    """
    # Exit early, if one of the lists or both are empty.
    if not assigned or not extracted:
        return 0.0

    # Store the longest keyphrases first.
    assigned_sets = sorted([set(keyword.split()) for keyword in assigned], key = len, reverse = True)
    extracted_sets = sorted([set(keyword.split()) for keyword in extracted], key = len, reverse = True)

    # This list stores True, if the assigned keyphrase has been matched earlier.
    # To avoid counting duplicate matches.
    assigned_matches = [False for assigned_set in assigned_sets]

    # For each extracted keyphrase, find the closest match, 
    # which is the assigned keyphrase it has the most words in common.
    for extracted_set in extracted_sets:
        all_matches = [(i, len(assigned_set & extracted_set)) for i, assigned_set in enumerate(assigned_sets)]
        closest_match = sorted(all_matches, key = lambda x: x[1], reverse = True)[0]
        assigned_matches[closest_match[0]] = True

    # Calculate the precision and recall metrics based on the partial matches.
    partial_matches = assigned_matches.count(True)  
    precision_k = partial_matches / k
    recall_k = partial_matches / len(assigned)
    
    return (
        2 * precision_k * recall_k / (precision_k + recall_k)
        if precision_k and recall_k else 0.0
    )


def f1_metric_k(assigned, extracted, k, partial_match = True):
    """
    Wrapper function that calculates either the exact
    or the partial match f1 metric.
    """
    return (
        partial_f1_k(assigned, extracted, k) 
        if partial_match else exact_f1_k(assigned, extracted, k)
    )
