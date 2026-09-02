<script lang="ts">
    import { BaseLayout } from '$lib/components/custom/method_layout';
    import VisualizationsPanel from '$lib/components/custom/visualizations-panel/visualizations-panel.svelte';
    import CandidateCard from './CandidateCard.svelte';
    import { 
        initializeFavoriteSession, 
        getFavoriteState, 
        submitFavoriteVote, 
        iterateFavoriteSession 
    } from './handler';
    import type { FavoriteSessionState } from './types';
    import { onMount, onDestroy } from 'svelte';

    let { data } = $props();

    // UI State
    let currentRole = $state<string>("analyst");
    let session = $state<FavoriteSessionState | null>(null);
    let selectedProblemId = $state<number>(1);

    $effect(() => {
        if (data.problems?.length && !session) {
            selectedProblemId = data.problems[0].id;
        }
    });
    let isProcessing = $state(false);
    let pollInterval: ReturnType<typeof setInterval>;

    let candidateObjectives = $derived<number[][]>(
        session?.candidates
            ? session.candidates.map((c) => Object.values(c.objective_values ?? {}))
            : []
    );

    // Available DMs: automatically derived from active session, or from group info, or default
    let groupDms = $derived<string[]>(
        session?.dm_ids ??
        (data.group?.user_ids?.length 
            ? data.group.user_ids.map((_: any, i: number) => `dm${i + 1}`)
            : ["dm1", "dm2", "dm3"])
    );

    let isDecisionPhase = $derived(
        session?.phase === "decision"
    );

    let currentProblem = $derived(
        data.problems?.find((p: any) => Number(p.id) === Number(selectedProblemId))
    );

    function startPolling() {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(async () => {
            if (session?.session_id && session.status !== "completed") {
                const updatedState = await getFavoriteState(session.session_id);
                if (updatedState) session = updatedState;
            }
        }, 3000);
    }

    onMount(() => startPolling());
    onDestroy(() => { if (pollInterval) clearInterval(pollInterval); });

    async function handleInit() {
        isProcessing = true;

        const probId = Number(selectedProblemId) || data.problems[0]?.id || 1;
        const active_dms = [...groupDms];
        let mps_payload: Record<string, Record<string, number>> = {};

        if (probId === 1) {
            // Problem 1: Discrete River Pollution (4 objectives - all maximized)
            // Pareto optimal solutions from datasets/river_poll_4_objs.csv
            const defaultMps: Record<string, Record<string, number>> = {
                "dm1": {"f1": 5.9066, "f2": 3.2894, "f3": 6.5792, "f4": -4.5460},
                "dm2": {"f1": 5.4290, "f2": 3.0121, "f3": 7.2395, "f4": -0.9135},
                "dm3": {"f1": 6.1623, "f2": 2.8839, "f3": 5.2575, "f4": -0.0045},
            };
            active_dms.forEach((dm, i) => {
                mps_payload[dm] = defaultMps[dm] || {
                    "f1": 5.80 + (i * 0.1), 
                    "f2": 3.10 + (i * 0.05), 
                    "f3": 6.00 + (i * 0.2), 
                    "f4": -3.00 + (i * 0.5)
                };
            });
        } else {
            // Problem 2: DTLZ2 (3 objectives)
            const defaultMps: Record<string, Record<string, number>> = {
                "dm1": {"f_1": 0.6666, "f_2": 0.6666, "f_3": 0.3333},
                "dm2": {"f_1": 0.6666, "f_2": 0.3333, "f_3": 0.6666},
                "dm3": {"f_1": 0.3333, "f_2": 0.6666, "f_3": 0.6666},
            };
            active_dms.forEach((dm, i) => {
                mps_payload[dm] = defaultMps[dm] || {
                    "f_1": 0.5 + (i * 0.1),
                    "f_2": 0.5 - (i * 0.05),
                    "f_3": 0.5
                };
            });
        }

        const newState = await initializeFavoriteSession({
            problem_id: probId,
            dm_ids: active_dms,
            total_n_of_candidates: 5,
            candidate_generation_options: "mm",
            max_iterations: 3,
            num_initial_reference_points: 50,
            most_preferred_solutions: mps_payload
        });

        if (newState) {
            session = newState;
            startPolling();
        }
        isProcessing = false;
    }

    async function handleVote(candidateIndex: number) {
        if (!session || currentRole === "analyst") return;
        isProcessing = true;
        const res = await submitFavoriteVote(session.session_id, {
            dm_id: currentRole,
            vote_idx: candidateIndex
        });
        if (res) {
            session.current_votes = { ...res.current_votes };
            if (res.status) session.status = res.status;
            if (res.phase) session.phase = res.phase;
            if (res.tie_state !== undefined) session.tie_state = res.tie_state;
            if (res.final_solution) session.final_solution = res.final_solution;

            const updated = await getFavoriteState(session.session_id);
            if (updated) session = updated;
        }
        isProcessing = false;
    }

    async function handleIterate() {
        if (!session) return;
        isProcessing = true;
        const newState = await iterateFavoriteSession(session.session_id);
        if (newState) session = newState;
        isProcessing = false;
    }
</script>

<!-- Prototype Role Simulator Bar: automatically matches the number of DMs in the group -->
<div class="bg-gray-800 text-white p-2 flex justify-center gap-4 text-sm z-50 relative flex-wrap items-center">
    <span class="font-bold">Prototype Role Simulator:</span>
    <label class="cursor-pointer flex items-center gap-1">
        <input type="radio" name="role" checked={currentRole === 'analyst'} onchange={() => currentRole = 'analyst'}> Analyst
    </label>
    {#each groupDms as dm, i}
        <label class="cursor-pointer flex items-center gap-1">
            <input type="radio" name="role" checked={currentRole === dm} onchange={() => currentRole = dm}> DM {i + 1} ({dm.toUpperCase()})
        </label>
    {/each}
</div>

<!-- Phase Indicator Bar: explicitly states "Decision Phase" or "Consensus-Reaching Phase" under Role Simulator -->
<div class="bg-slate-100 border-b px-4 py-2 flex justify-between items-center text-sm shadow-sm">
    <div class="flex items-center gap-3">
        {#if session}
            {#if session.status === "completed"}
                <span class="inline-flex items-center gap-1 px-3 py-0.5 rounded-full text-xs font-bold uppercase tracking-wide bg-green-200 text-green-900 border border-green-400">
                    🏆 Decision Phase Complete
                </span>
                <span class="text-sm font-semibold text-gray-800">Group Decision Complete! Winning Solution Selected</span>
            {:else if isDecisionPhase}
                <span class="inline-flex items-center gap-1 px-3 py-0.5 rounded-full text-xs font-bold uppercase tracking-wide bg-amber-200 text-amber-900 border border-amber-400">
                    🎯 Decision Phase
                </span>
                <span class="text-sm font-semibold text-gray-800">Final Decision Phase (Final Voting)</span>
            {:else}
                <span class="inline-flex items-center gap-1 px-3 py-0.5 rounded-full text-xs font-bold uppercase tracking-wide bg-blue-100 text-blue-900 border border-blue-300">
                    🤝 Consensus-Reaching Phase
                </span>
                <span class="text-sm font-semibold text-gray-800">Iteration {session.current_iteration} of {session.max_iterations} (Zooming in on preferred solutions)</span>
            {/if}
        {:else}
            <span class="inline-flex items-center gap-1 px-3 py-0.5 rounded-full text-xs font-bold uppercase tracking-wide bg-gray-200 text-gray-700 border border-gray-300">
                ⚙️ Setup Phase
            </span>
            <span class="text-sm text-gray-600">Select a problem to initiate the FAVORITE session</span>
        {/if}
    </div>
    <div class="text-xs text-gray-500 font-medium">
        {#if data.group}
            Group: <strong class="text-gray-700">{data.group.name ?? 'Group ' + data.group.id}</strong> ({groupDms.length} DMs)
        {:else}
            Group: <strong class="text-gray-700">{groupDms.length} Decision Makers</strong>
        {/if}
    </div>
</div>

<BaseLayout showLeftSidebar={true} showRightSidebar={false}>
    
    {#snippet leftSidebar()}
        <div class="p-4 flex flex-col gap-6">
            <h2 class="text-xl font-bold border-b pb-2">Favorite Method</h2>

            {#if !session}
                {#if currentRole === "analyst"}
                    <div>
                        <label for="problem-select" class="block text-sm font-medium mb-1">Select Problem</label>
                        <select id="problem-select" bind:value={selectedProblemId} class="w-full border p-2 rounded">
                            {#each data.problems as prob}
                                <option value={prob.id}>{prob.name}</option>
                            {/each}
                        </select>
                    </div>
                    <div class="text-xs text-gray-500">
                        Configured with <strong>{groupDms.length} Decision Makers</strong>: {groupDms.join(", ")}
                    </div>
                    <button 
                        class="bg-green-600 text-white py-2 rounded hover:bg-green-700 disabled:opacity-50 font-semibold"
                        onclick={handleInit}
                        disabled={isProcessing}
                    >
                        {isProcessing ? 'Initializing Engine...' : 'Start Session'}
                    </button>
                {:else}
                    <p class="text-gray-500 italic">Waiting for Analyst to start the session...</p>
                {/if}

            {:else}
                <div class="bg-gray-50 p-3 rounded border">
                    <p class="text-sm">
                        <strong>Phase:</strong> 
                        <span class="font-bold text-indigo-700">
                            {session.status === 'completed' ? 'Completed' : isDecisionPhase ? 'Final Decision Phase' : 'Consensus-Reaching Phase'}
                        </span>
                    </p>
                    {#if session.status !== "completed"}
                        <p class="text-sm"><strong>Iteration:</strong> {session.current_iteration} / {session.max_iterations}</p>
                    {/if}
                    <p class="text-sm"><strong>Status:</strong> <span class="capitalize text-blue-600">{session.status.replace(/_/g, ' ')}</span></p>
                </div>

                <div>
                    <h3 class="font-bold mb-2">Voting Status</h3>
                    <ul class="text-sm space-y-1">
                        {#each session.dm_ids as dm}
                            <li>
                                {dm}: 
                                {#if session.current_votes[dm] !== undefined}
                                    <span class="text-green-600 font-medium">✅ Voted (Cand {session.current_votes[dm] + 1})</span>
                                {:else}
                                    <span class="text-orange-500 font-medium">⏳ Waiting...</span>
                                {/if}
                            </li>
                        {/each}
                    </ul>
                </div>

                {#if currentRole === "analyst"}
                    {#if session.status === "completed"}
                        <div class="bg-green-100 text-green-800 p-3 rounded font-bold text-center border border-green-300">
                            🎉 Group Decision Complete!
                        </div>
                    {:else if isDecisionPhase}
                        <div class="bg-indigo-50 border border-indigo-200 text-indigo-900 p-3 rounded text-sm text-center">
                            <span class="font-bold block mb-1">Final Decision Phase</span>
                            <p class="text-xs text-indigo-700">Waiting for all Decision Makers to cast their final vote.</p>
                        </div>
                    {:else}
                        {@const allVoted = Object.keys(session.current_votes).length === session.dm_ids.length}
                        <button 
                            class="bg-purple-600 text-white py-2 rounded hover:bg-purple-700 disabled:opacity-50 font-semibold shadow"
                            onclick={handleIterate}
                            disabled={isProcessing || !allVoted || session.status === "revote_pending"}
                        >
                            {isProcessing ? 'Computing...' : 'Run Next Iteration'}
                        </button>
                        {#if !allVoted}
                            <p class="text-xs text-gray-500 text-center">Waiting for all votes...</p>
                        {/if}
                    {/if}
                {/if}
            {/if}
        </div>
    {/snippet}

    {#snippet visualizationArea()}
        {#if session && currentProblem}
            <div class="p-4 flex flex-col gap-6 h-full overflow-y-auto">
                <div class="h-64 border rounded shadow-sm bg-white p-2">
                    <VisualizationsPanel 
                        problem={currentProblem} 
                        solutionsObjectiveValues={candidateObjectives} 
                        previousPreferenceType="reference_point"
                        currentPreferenceType="reference_point"
                    />
                </div>

                {#if session.status === "completed"}
                    <div class="bg-green-100 text-green-900 p-4 rounded border border-green-400 font-medium text-center shadow-sm">
                        <span class="font-bold text-base">🎉 Group Decision Complete! Winning Solution Selected.</span>
                        <p class="text-sm mt-1">Review the selected final solution below or switch to the Data Table to inspect values.</p>
                    </div>
                {:else if isDecisionPhase}
                    <div class="bg-indigo-50 border border-indigo-200 text-indigo-900 p-3 rounded text-sm shadow-sm flex items-center justify-between">
                        <div>
                            <strong>Final Iteration Reached:</strong> Cast your final vote for the group's consensus solution.
                        </div>
                        <span class="text-xs font-bold bg-indigo-200 text-indigo-900 px-2 py-1 rounded">Final Vote</span>
                    </div>
                {/if}

                {#if session.status === "revote_pending"}
                    {@const tiedIndices = session.tie_state?.tied_candidate_indices ?? []}
                    {@const tiedListStr = tiedIndices.map((idx: number) => `Candidate ${idx + 1}`).join(", ")}
                    <div class="bg-amber-100 text-amber-900 p-4 rounded border border-amber-400 font-medium text-center shadow-sm">
                        <span class="font-bold text-base">⚠️ Tie detected between {tiedListStr}!</span>
                        <p class="text-sm mt-1">Decision Makers must cast a revote between the tied candidates.</p>
                    </div>
                {/if}

                <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4">
                    {#each session.candidates as candidate, i}
                        {@const tiedIndices = session.tie_state?.tied_candidate_indices ?? []}
                        {@const isRevote = session.status === "revote_pending"}
                        {@const isTied = isRevote && tiedIndices.includes(i)}
                        {@const hasCurrentRoleVoted = session.current_votes[currentRole] !== undefined}
                        {@const isCardDisabled = isProcessing || (isRevote ? (!isTied || hasCurrentRoleVoted) : (hasCurrentRoleVoted && session.current_votes[currentRole] !== i))}
                        {@const isFinalSolution = session.status === "completed" && (
                            session.final_solution
                                ? JSON.stringify(session.final_solution.objective_values) === JSON.stringify(candidate.objective_values)
                                : session.tie_state?.final_winner_idx === i
                        )}
                        <CandidateCard 
                            {candidate} 
                            index={i} 
                            onVote={handleVote}
                            showVoteButton={currentRole !== "analyst" && session.status !== "completed"}
                            isVoted={session.current_votes[currentRole] === i}
                            disabled={isCardDisabled}
                            isTiedCandidate={isTied}
                            isDecisionPhase={isDecisionPhase}
                            isFinalWinner={isFinalSolution}
                        />
                    {/each}
                </div>
            </div>
        {/if}
    {/snippet}

    {#snippet numericalValues()}
        {#if session && session.status === "completed" && session.final_solution}
            <div class="p-4 bg-white rounded h-full overflow-y-auto">
                <div class="bg-green-50 border border-green-300 rounded p-4 mb-4 shadow-sm">
                    <h3 class="font-bold text-lg text-green-900 flex items-center gap-2">
                        👑 Winning Consensus Solution
                    </h3>
                    <p class="text-sm text-green-800 mt-1">
                        Fairness Criterion: <span class="font-semibold">{session.final_solution.fairness_criterion}</span>
                        {#if session.final_solution.fairness_value !== undefined && session.final_solution.fairness_value < 1000}
                            (Score: {session.final_solution.fairness_value.toFixed(4)})
                        {/if}
                    </p>
                </div>
                <h4 class="font-bold mb-2 text-md text-gray-800">Exact Objective Values</h4>
                <table class="w-full text-sm text-left border-collapse mb-6 border">
                    <thead>
                        <tr class="border-b-2 border-gray-300 bg-gray-50">
                            <th class="p-2.5 font-bold text-gray-700">Objective</th>
                            <th class="p-2.5 font-bold text-gray-700">Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {#each Object.entries(session.final_solution.objective_values ?? {}) as [key, val]}
                            <tr class="border-b last:border-0 hover:bg-gray-50">
                                <td class="p-2.5 font-medium text-gray-700">{key}</td>
                                <td class="p-2.5 font-mono text-gray-900 font-semibold">{Number(val).toFixed(4)}</td>
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        {:else if session && session.candidates.length > 0}
            <div class="p-4 bg-white rounded h-full overflow-y-auto">
                <h3 class="font-bold mb-3 text-lg">Candidate Data Table</h3>
                <table class="w-full text-sm text-left border-collapse">
                    <thead>
                        <tr class="border-b-2 border-gray-300">
                            <th class="pb-2 pr-4 font-bold text-gray-700">Candidate</th>
                            <th class="pb-2 pr-4 font-bold text-gray-700">Fairness</th>
                            {#each Object.keys(session.candidates[0].objective_values ?? {}) as objKey}
                                <th class="pb-2 pr-4 font-bold text-gray-700">{objKey}</th>
                            {/each}
                        </tr>
                    </thead>
                    <tbody>
                        {#each session.candidates as cand, i}
                            <tr class="border-b last:border-0 hover:bg-gray-50">
                                <td class="py-3 pr-4 font-semibold">Candidate {i + 1}</td>
                                <td class="py-3 pr-4 text-gray-600">
                                    {#if cand.fairness_criterion.includes("hausdorff")}
                                        Diversity
                                    {:else}
                                        {cand.fairness_criterion} <br/>
                                        <span class="text-xs">({cand.fairness_value.toFixed(4)})</span>
                                    {/if}
                                </td>
                                {#each Object.values(cand.objective_values ?? {}) as val}
                                    <td class="py-3 pr-4 font-mono">{Number(val).toFixed(4)}</td>
                                {/each}
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        {:else}
            <div class="flex h-full items-center justify-center text-gray-500 italic">
                No candidates generated yet. Start the session to view data.
            </div>
        {/if}
    {/snippet}

</BaseLayout>
