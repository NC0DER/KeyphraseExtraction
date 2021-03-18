def precision_k(assigned, extracted, k):
    """
    Computes the exact match precision at k, 
    between two lists of keywords. 
    The precision is defined as the fraction
    between the number of correctly matched tokens
    (the intersection of assigned and extracted sets)
    over the number of extracted (k) tokens.
    Arguments
    ---------
    assigned  : A list of manually assigned keywords,
                (order doesn't matter in the list).
    extracted : A list of extracted keywords,
                (order matters in the list).
    k         : int
                The maximum number of extracted keywords.
    Returned value
    --------------
              : double
    """
    return len(set(assigned) & set(extracted[:k])) / k

def recall_k(assigned, extracted, k):
    """
    Computes the exact match recall at k,
    between two lists of keywords.
    The average precision is defined as the fraction 
    between the number of correctly matched tokens 
    (the intersection of assigned and extracted sets)
    over the number of assigned tokens.
    Arguments
    ---------
    assigned  : A list of manually assigned keywords,
                (order doesn't matter in the list).
    extracted : A list of extracted keywords,
                (order matters in the list).
    k         : int
                The maximum number of extracted keywords.
    Returned value
    --------------
              : double
    """
    return len(set(assigned) & set(extracted[:k])) / len(assigned)

def partial_precision_k(assigned, extracted, k):
    """
    Computes the average partial precision at k, 
    between two lists of keywords.
    The partial precision is defined as the fraction 
    between the number of correctly partially matched tokens,
    over the total number of extracted (k) tokens.
    Arguments
    ---------
    assigned  : A list of manually assigned keywords,
                (order doesn't matter in the list).
    extracted : A list of extracted keywords,
                (order matters in the list).
    k         : int
                The maximum number of extracted keywords.
    Returned value
    --------------
              : double
    """
    # Assigned should always contain the shorter list, while extracted the longest,
    # as to avoid counting partial matches more times than necessary.
    assigned, extracted = min((assigned, extracted[:k]), key = len), max((assigned, extracted[:k]), key = len)
    assigned_sets = [set(keyword.split()) for keyword in assigned]
    extracted_sets = [set(keyword.split()) for keyword in extracted]

    return sum(
        1.0 for i in assigned_sets  
            if any(True for j in extracted_sets if i & j)) / k

def partial_recall_k(assigned, extracted, k):
    """
    Computes the average partial recall at k, 
    between two lists of keywords.
    The partial recall is defined as the fraction 
    between the number of correctly partially matched tokens,
    over the total number of extracted (k) tokens.
    Arguments
    ---------
    assigned  : A list of manually assigned keywords,
                (order doesn't matter in the list).
    extracted : A list of extracted keywords,
                (order matters in the list).
    k         : int
                The maximum number of extracted keywords.
    Returned value
    --------------
              : double
    """
    # Assigned should always contain the shorter list, while extracted the longest,
    # as to avoid counting partial matches more times than necessary.
    assigned_length = len(assigned)
    assigned, extracted = min((assigned, extracted[:k]), key = len), max((assigned, extracted[:k]), key = len)
    assigned_sets = [set(keyword.split()) for keyword in assigned]
    extracted_sets = [set(keyword.split()) for keyword in extracted]

    return sum(
        1.0 for i in assigned_sets
            if any(True for j in extracted_sets if i & j)) / assigned_length

def f1_measure_k(assigned, extracted, k, partial):
    """
    Computes the f1 measure at k.
    The f1 measure at k is defined as 
    the harmonic mean of the
    precision at k and recall at k.
    Arguments
    ---------
    assigned  : A list of keywords that are to be extracted,
                (order doesn't matter in the list).
    extracted : A list of lists of extracted keywords,
                (order matters in the list).
    k         : int
                The maximum number of extracted keywords.
    partial   : boolean
                If set to True, partial matching
                f1 measure at k is calculated.
    Returned value
    --------------
              : double
    """
    precision = (
        partial_precision_k(assigned, extracted, k)
        if partial else precision_k(assigned, extracted, k)
    )
    recall = (
        partial_recall_k(assigned, extracted, k)
        if partial else recall_k(assigned, extracted, k)
    )
    return (
        2 * precision * recall / (precision + recall)
        if not precision == recall == 0.0 else 0.0
    )

def main():
    assigned = ['green big deal', 'green big energy', 'forest', 'smoke', 'mirrors']
    extracted = ['green horse', 'green small energy', 'wind']
    print(partial_precision_k(assigned, extracted, 3))
    print(partial_recall_k(assigned, extracted, 3))
    print(f1_measure_k(assigned, extracted, 3, partial = True))

if __name__ == '__main__': main()