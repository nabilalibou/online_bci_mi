def add_prefix_to_list(list_str, prefix):
    return list(map(lambda x: prefix + x, list_str))


def add_suffix_to_list(list_str, suffix):
    return list(map(lambda x: x + suffix, list_str))
