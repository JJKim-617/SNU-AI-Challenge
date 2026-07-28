from typing import Iterable, Tuple

Permutation = Tuple[int, int, int, int]


def validate_permutation(values: Iterable[int]) -> Permutation:
    result = tuple(int(value) for value in values)
    if len(result) != 4 or sorted(result) != [1, 2, 3, 4]:
        raise ValueError(f"Expected a permutation of 1..4, got {result}")
    return result  # type: ignore[return-value]


def ranks_to_order(ranks: Iterable[int]) -> Permutation:
    checked = validate_permutation(ranks)
    return validate_permutation(
        input_index
        for _, input_index in sorted(
            (rank, index + 1) for index, rank in enumerate(checked)
        )
    )


def order_to_ranks(order: Iterable[int]) -> Permutation:
    checked = validate_permutation(order)
    ranks = [0, 0, 0, 0]
    for chronological_position, input_index in enumerate(checked, start=1):
        ranks[input_index - 1] = chronological_position
    return validate_permutation(ranks)


def reorder_ranks(
    ranks: Iterable[int], new_input_order: Iterable[int]
) -> Permutation:
    checked_ranks = validate_permutation(ranks)
    checked_order = validate_permutation(new_input_order)
    return validate_permutation(
        checked_ranks[old_index - 1] for old_index in checked_order
    )


def pairwise_precedence(
    ranks: Iterable[int],
) -> Tuple[int, int, int, int, int, int]:
    checked = validate_permutation(ranks)
    return tuple(
        int(checked[left] < checked[right])
        for left in range(4)
        for right in range(left + 1, 4)
    )  # type: ignore[return-value]
