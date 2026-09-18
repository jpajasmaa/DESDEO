"""This module contains voting rules for group decision making such as majority rule."""

from collections import Counter


def majority_rule(votes: dict[str, int]) -> int | None:
    """Choose the option that has more than half of the votes.

    Args:
        votes (dict[str, int]): A dictionary mapping voter IDs to their votes.

    Returns:
        int | None: The option that has more than half of the votes, or None if no such option exists.
    """
    counts = Counter(votes.values())
    all_votes = sum(counts.values())
    for vote, c in counts.items():
        if c > all_votes // 2:
            return vote
    return None


def plurality_rule(votes: dict[str, int]) -> list[int]:
    """Choose the option that has the most votes.

    Args:
        votes (dict[str, int]): A dictionary mapping voter IDs to their votes.

    Returns:
        list[int]: A list of options that have the most votes (in case of a tie).
    """
    counts = Counter(votes.values())
    max_votes = max(counts.values())
    return [vote for vote, c in counts.items() if c == max_votes]


def consensus_rule(votes: dict[str, int], min_votes: int) -> list[int]:
    """Choose all options that have at least min_votes votes.

    Args:
        votes (dict[str, int]): A dictionary mapping voter IDs to their votes.
        min_votes (int): The minimum number of votes required for an option to be selected.

    """
    if min_votes <= 0:
        raise ValueError("min_votes must be greater than 0.")
    if min_votes > len(votes):
        raise ValueError("min_votes cannot be greater than the number of voters.")
    counts = Counter(votes.values())
    return [vote for vote, c in counts.items() if c >= min_votes]


def calculate_borda_scores(
    round_1_votes: dict[str, int],
    round_2_votes: dict[str, int],
    n_candidates: int,
    weights: tuple[int | float, int | float] = (2, 1),
) -> dict[int, int | float]:
    """Calculate weighted Borda scores across Round 1 and Round 2 votes.

    Each candidate receives w_1 points per Round 1 vote and w_2 points per Round 2 vote:
        Score(c) = weights[0] * V_1(c) + weights[1] * V_2(c)

    Args:
        round_1_votes: Map of DM ID to candidate index chosen in Round 1.
        round_2_votes: Map of DM ID to candidate index chosen in Round 2.
        n_candidates: Total number of candidates.
        weights: Tuple of weights (round_1_weight, round_2_weight). Defaults to (2, 1).

    Returns:
        dict[int, int | float]: Map of candidate index to calculated Borda score.
    """
    scores: dict[int, int | float] = dict.fromkeys(range(n_candidates), 0)
    w1, w2 = weights
    for cand in round_1_votes.values():
        if 0 <= cand < n_candidates:
            scores[cand] += w1
    for cand in round_2_votes.values():
        if 0 <= cand < n_candidates:
            scores[cand] += w2
    return scores


def borda_rule(
    round_1_votes: dict[str, int],
    round_2_votes: dict[str, int],
    n_candidates: int,
    weights: tuple[int | float, int | float] = (2, 1),
) -> list[int]:
    """Find the candidate index/indices with the highest weighted Borda score.

    Args:
        round_1_votes: Map of DM ID to candidate index chosen in Round 1.
        round_2_votes: Map of DM ID to candidate index chosen in Round 2.
        n_candidates: Total number of candidates.
        weights: Tuple of weights (round_1_weight, round_2_weight). Defaults to (2, 1).

    Returns:
        list[int]: Candidate indices achieving the maximum Borda score (length 1 if unique winner).
    """
    scores = calculate_borda_scores(round_1_votes, round_2_votes, n_candidates, weights=weights)
    max_score = max(scores.values()) if scores else 0
    return [cand for cand, score in scores.items() if score == max_score]

