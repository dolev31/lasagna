import lasagna


@lasagna.reranker
def keyword_overlap(query, text):  # my technique: query words found in the passage
    return len(set(query.lower().split()) & set(text.lower().split()))
