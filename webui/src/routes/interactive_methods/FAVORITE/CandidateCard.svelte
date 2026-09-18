<script lang="ts">
    import type { FairSolution } from './types';

    let {
        candidate,
        index,
        onVote,
        disabled = false,
        isVoted = false,
        showVoteButton = false,
        isTiedCandidate = false,
        isDecisionPhase = false,
        isFinalWinner = false,
        isIterationWinner = false,
        isExcludedFromRevote = false,
        isRevotePhase = false,
        bordaScore = null,
        objectiveNameMap = {}
    }: {
        candidate: FairSolution;
        index: number;
        onVote: (idx: number) => void;
        disabled?: boolean;
        isVoted?: boolean;
        showVoteButton?: boolean;
        isTiedCandidate?: boolean;
        isDecisionPhase?: boolean;
        isFinalWinner?: boolean;
        isIterationWinner?: boolean;
        isExcludedFromRevote?: boolean;
        isRevotePhase?: boolean;
        bordaScore?: number | null;
        objectiveNameMap?: Record<string, string>;
    } = $props();

    // Extract objective keys and values for clean rendering using standard objective_values
    let objectives = $derived(Object.entries(candidate.objective_values ?? {}));
</script>

<div class="card {isFinalWinner ? 'border-yellow-500 bg-yellow-50/60 ring-2 ring-yellow-400' : isIterationWinner ? 'border-emerald-500 bg-emerald-50/40 ring-2 ring-emerald-400' : isExcludedFromRevote ? 'border-rose-200 bg-rose-50/30' : isTiedCandidate ? 'border-amber-500 bg-amber-50/40 ring-2 ring-amber-400' : isVoted ? 'border-green-500 bg-green-50' : 'border-gray-200'} border rounded-lg p-4 shadow-sm flex flex-col h-full transition-all">
    <div class="mb-3 border-b pb-2 flex justify-between items-start">
        <div>
            <h3 class="text-lg font-bold flex items-center gap-1.5">
                Candidate {index + 1}
                {#if isFinalWinner}
                    <span class="text-xs bg-yellow-200 text-yellow-900 font-bold px-2 py-0.5 rounded">👑 Final Choice</span>
                {:else if isIterationWinner}
                    <span class="text-xs bg-emerald-200 text-emerald-900 font-bold px-2 py-0.5 rounded">⭐ Iteration Winner</span>
                {/if}
            </h3>
            <p class="text-xs text-gray-500">
                {#if candidate.fairness_criterion === "last_winner"}
                    Criterion: Last Iteration Winner
                {:else if candidate.fairness_criterion === "final_hausdorff" || candidate.fairness_criterion === "avg_hausdorff"}
                    Criterion: Diversity (Hausdorff)
                {:else if candidate.fairness_criterion?.startsWith("winner_and_")}
                    Criterion: Group Fair & Last Voted (Duplicate Merged)
                {:else if candidate.fairness_criterion?.startsWith("final_")}
                    Criterion: Group Fair ({candidate.fairness_criterion.replace("final_", "").toUpperCase()})
                {:else}
                    Fairness: {candidate.fairness_criterion} ({candidate.fairness_value.toFixed(4)})
                {/if}
            </p>
        </div>
        <div class="flex flex-col gap-1 items-end">
            {#if bordaScore !== null && bordaScore !== undefined}
                <span class="text-xs bg-indigo-100 text-indigo-900 font-bold px-2 py-0.5 rounded border border-indigo-200 shadow-sm">
                    Borda: {bordaScore} pts
                </span>
            {/if}
            {#if isExcludedFromRevote}
                <span class="text-xs bg-rose-100 text-rose-800 font-semibold px-2 py-0.5 rounded border border-rose-200">
                    1st Choice (Concession Required)
                </span>
            {:else if isTiedCandidate}
                <span class="text-xs bg-amber-200 text-amber-900 font-semibold px-2 py-0.5 rounded">Tied</span>
            {/if}
            {#if isDecisionPhase}
                {#if candidate.fairness_criterion === "last_winner"}
                    <span class="text-xs bg-purple-100 text-purple-800 font-semibold px-1.5 py-0.5 rounded">Last Voted</span>
                {:else if candidate.fairness_criterion?.startsWith("winner_and_")}
                    <span class="text-xs bg-indigo-100 text-indigo-800 font-semibold px-1.5 py-0.5 rounded" title="The group-fair solution was identical to the previous winner, so they have been merged. An extra diversity candidate was added.">Fair & Last Voted (Merged)</span>
                {:else if candidate.fairness_criterion?.startsWith("final_") && candidate.fairness_criterion !== "final_hausdorff"}
                    <span class="text-xs bg-blue-100 text-blue-800 font-semibold px-1.5 py-0.5 rounded">Fair Solution</span>
                {:else if candidate.fairness_criterion === "final_hausdorff" || candidate.fairness_criterion === "avg_hausdorff"}
                    <span class="text-xs bg-teal-100 text-teal-800 font-semibold px-1.5 py-0.5 rounded">Diversity</span>
                {/if}
            {:else}
                {#if candidate.fairness_criterion?.startsWith("winner_and_")}
                    <span class="text-xs bg-indigo-100 text-indigo-800 font-semibold px-1.5 py-0.5 rounded" title="The group-fair solution was identical to the previous winner, so they have been merged. An extra diversity candidate was added.">Fair & Last Voted (Merged)</span>
                {/if}
            {/if}
        </div>
    </div>

    <div class="flex-grow mb-4">
        <ul class="text-sm space-y-1">
            {#each objectives as [key, val]}
                <li class="flex justify-between">
                    <span class="font-medium text-gray-600">{objectiveNameMap[key] || key}:</span>
                    <span>{Number(val).toFixed(4)}</span>
                </li>
            {/each}
        </ul>
    </div>

    {#if showVoteButton}
        <button
            class="w-full py-2 rounded font-semibold text-white transition-colors {isVoted ? 'bg-green-600' : isExcludedFromRevote ? 'bg-gray-400' : isRevotePhase ? 'bg-amber-600 hover:bg-amber-700' : isDecisionPhase ? 'bg-indigo-600 hover:bg-indigo-700' : 'bg-blue-600 hover:bg-blue-700'} disabled:opacity-50 disabled:cursor-not-allowed"
            onclick={() => onVote(index)}
            disabled={disabled || isVoted || isExcludedFromRevote}
        >
            {isVoted ? '✓ Your Choice' : isExcludedFromRevote ? '1st Choice (Ineligible)' : isRevotePhase ? 'Vote as Concession' : isDecisionPhase ? 'Vote as Final Choice' : 'Vote as Favorite'}
        </button>
    {/if}
</div>
