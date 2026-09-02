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
        isFinalWinner = false
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
    } = $props();

    // Extract objective keys and values for clean rendering using standard objective_values
    let objectives = $derived(Object.entries(candidate.objective_values ?? {}));
</script>

<div class="card {isFinalWinner ? 'border-yellow-500 bg-yellow-50/60 ring-2 ring-yellow-400' : isTiedCandidate ? 'border-amber-500 bg-amber-50/40 ring-2 ring-amber-400' : isVoted ? 'border-green-500 bg-green-50' : 'border-gray-200'} border rounded-lg p-4 shadow-sm flex flex-col h-full transition-all">
    <div class="mb-3 border-b pb-2 flex justify-between items-start">
        <div>
            <h3 class="text-lg font-bold flex items-center gap-1.5">
                Candidate {index + 1}
                {#if isFinalWinner}
                    <span class="text-xs bg-yellow-200 text-yellow-900 font-bold px-2 py-0.5 rounded">👑 Final Choice</span>
                {/if}
            </h3>
            <p class="text-xs text-gray-500">
                {#if candidate.fairness_criterion === "last_winner"}
                    Criterion: Last Iteration Winner
                {:else if candidate.fairness_criterion === "final_hausdorff" || candidate.fairness_criterion === "avg_hausdorff"}
                    Criterion: Diversity (Hausdorff)
                {:else if candidate.fairness_criterion?.startsWith("final_")}
                    Criterion: Group Fair ({candidate.fairness_criterion.replace("final_", "").toUpperCase()})
                {:else}
                    Fairness: {candidate.fairness_criterion} ({candidate.fairness_value.toFixed(4)})
                {/if}
            </p>
        </div>
        <div class="flex flex-col gap-1 items-end">
            {#if isTiedCandidate}
                <span class="text-xs bg-amber-200 text-amber-900 font-semibold px-2 py-0.5 rounded">Tied</span>
            {/if}
            {#if isDecisionPhase}
                {#if candidate.fairness_criterion === "last_winner"}
                    <span class="text-xs bg-purple-100 text-purple-800 font-semibold px-1.5 py-0.5 rounded">Last Voted</span>
                {:else if candidate.fairness_criterion?.startsWith("final_") && candidate.fairness_criterion !== "final_hausdorff"}
                    <span class="text-xs bg-blue-100 text-blue-800 font-semibold px-1.5 py-0.5 rounded">Fair Solution</span>
                {:else if candidate.fairness_criterion === "final_hausdorff" || candidate.fairness_criterion === "avg_hausdorff"}
                    <span class="text-xs bg-teal-100 text-teal-800 font-semibold px-1.5 py-0.5 rounded">Diversity</span>
                {/if}
            {/if}
        </div>
    </div>

    <div class="flex-grow mb-4">
        <ul class="text-sm space-y-1">
            {#each objectives as [key, val]}
                <li class="flex justify-between">
                    <span class="font-medium text-gray-600">{key}:</span>
                    <span>{Number(val).toFixed(4)}</span>
                </li>
            {/each}
        </ul>
    </div>

    {#if showVoteButton}
        <button 
            class="w-full py-2 rounded font-semibold text-white transition-colors {isVoted ? 'bg-green-600' : isTiedCandidate ? 'bg-amber-600 hover:bg-amber-700' : 'bg-blue-600 hover:bg-blue-700'} disabled:opacity-50 disabled:cursor-not-allowed"
            onclick={() => onVote(index)}
            disabled={disabled || isVoted}
        >
            {isVoted ? '✓ Your Choice' : isTiedCandidate ? 'Vote in Revote' : isDecisionPhase ? 'Vote as Final Choice' : 'Vote as Favorite'}
        </button>
    {/if}
</div>
