"""Integer apportionment for a composition-preserving bounded chain graph."""

from typing import Optional, Sequence


def rescale_repeat_counts(counts: Sequence[int], max_tokens: Optional[int] = 1536) -> list[int]:
    """Rescale counts before expansion, with a fixed total repeat-token budget.

    Counts at or below the budget are unchanged; None or a nonpositive budget
    disables rescaling. Above the budget, quotas are ``budget * count / total``.
    Integer counts minimize squared deviation from those quotas subject to a
    fixed total and at least one repeat for every positive input segment. This
    is largest-remainder rounding when the positivity constraint is inactive.
    Ties follow input segment order, without consuming the random generator.

    The guarantee is proportional allocation up to integer rounding, not exact
    fractions for ratios that the integer budget cannot represent. Each positive
    block/monomer segment survives. A budget smaller than their number is rejected
    because it cannot represent that composition.
    """
    values = [int(n) for n in counts]
    if any(n < 0 or n != original for n, original in zip(values, counts)):
        raise ValueError("Repeat counts must be nonnegative integers.")
    total = sum(values)
    if max_tokens is None or max_tokens <= 0 or total <= max_tokens:
        return values
    budget = int(max_tokens)
    if budget != max_tokens:
        raise ValueError("The repeat-token budget must be an integer.")
    active = [i for i, n in enumerate(values) if n > 0]
    if budget < len(active):
        raise ValueError(
            f"max_chain_tokens={budget} cannot retain all {len(active)} "
            "positive block/monomer segments; increase the budget."
        )
    # Integer arithmetic makes quota ties deterministic even for very long chains.
    allocated = [max(1, n * budget // total) if n else 0 for n in values]
    remaining = budget - sum(allocated)
    while remaining > 0:
        index = max(active, key=lambda i: (values[i] * budget - allocated[i] * total, -i))
        allocated[index] += 1
        remaining -= 1
    while remaining < 0:
        candidates = [i for i in active if allocated[i] > 1]
        index = max(candidates, key=lambda i: (allocated[i] * total - values[i] * budget, -i))
        allocated[index] -= 1
        remaining += 1
    return allocated
