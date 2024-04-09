from typing import Dict, List


def add_prefix_to_list(list_str, prefix):
    return list(map(lambda x: prefix + x, list_str))


def add_suffix_to_list(list_str, suffix):
    return list(map(lambda x: x + suffix, list_str))


def auto_weight_chan_dict(ch_names: List[str]) -> Dict[str, float]:
    """
    Assigns weights to EEG channel names based on the 10-20 system.
    Weight values:
        - 0.5: Low importance (temporal, frontal-temporal, frontal-polar)
        - 1.0: Neutral importance (anterior frontal, posterior occipital, parieto-occipital)
        - 2.0: High importance (central, parietal, frontal, other numbered)

    Args:
        ch_names: List of EEG channel names (strings).

    Returns:
        Dictionary mapping channel names to their assigned weights.
    """
    weight_mapping = {
        "low": (("T", "FT", "Fp"), ("9", "10")),
        "neutral": (("AF", "PO", "O"), ("7", "8")),
        "high": (("A", "F", "C", "P"), ("1", "2", "3", "4", "5", "6", "z")),
    }
    weight_dict = {}
    for ch in ch_names:
        for importance, (prefixes, digits) in weight_mapping.items():
            if any(ch.startswith(prefix) for prefix in prefixes) or any(ch.endswith(digit) for digit in digits):
                weight_dict[ch] = {"low": 0.5, "neutral": 1.0, "high": 2.0}[importance]
                break  # Exit the inner loop after finding a match

    return weight_dict
