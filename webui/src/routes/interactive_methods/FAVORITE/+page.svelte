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

    // Analyst View Mode: "candidates" (default) | "current_mps" | "original_mps"
    let analystViewMode = $state<"candidates" | "current_mps" | "original_mps">("candidates");

    // Enforce default candidates view for non-analysts
    let activeViewMode = $derived<"candidates" | "current_mps" | "original_mps">(
        currentRole === "analyst" ? analystViewMode : "candidates"
    );

    // DM Palette for up to 5 DMs
    const DM_PALETTE = [
        { stroke: "#7c3aed", fill: "#f5f3ff", text: "#6d28d9", border: "#c4b5fd", pill: "bg-purple-100 text-purple-800 border-purple-300" },
        { stroke: "#059669", fill: "#ecfdf5", text: "#047857", border: "#6ee7b7", pill: "bg-emerald-100 text-emerald-800 border-emerald-300" },
        { stroke: "#d97706", fill: "#fffbeb", text: "#b45309", border: "#fcd34d", pill: "bg-amber-100 text-amber-800 border-amber-300" },
        { stroke: "#e11d48", fill: "#fff1f2", text: "#be123c", border: "#fda4af", pill: "bg-rose-100 text-rose-800 border-rose-300" },
        { stroke: "#0284c7", fill: "#f0f9ff", text: "#0369a1", border: "#7dd3fc", pill: "bg-sky-100 text-sky-800 border-sky-300" },
    ];
    function getDmColor(index: number) {
        return DM_PALETTE[index % DM_PALETTE.length];
    }

    // Maps for preferred solutions
    let originalMpsMap = $derived<Record<string, Record<string, number>>>(
        session?.options?.original_most_preferred_solutions ?? {}
    );
    let currentMpsMap = $derived<Record<string, Record<string, number>>>(
        session?.current_most_preferred_solutions ?? originalMpsMap
    );

    // Helpers to extract objective arrays in order of currentProblem.objectives
    function extractObjectiveArray(solObj: Record<string, number> | undefined): number[] {
        if (!solObj || !currentProblem?.objectives) return [];
        return currentProblem.objectives.map((obj: any) => solObj[obj.symbol] ?? 0);
    }

    // Helper to get user-friendly adjustment info for a DM
    function getDmAdjustmentInfo(dm: string): { adjusted: boolean; description: string; badgeClass: string } {
        if (!session?.mps_adjustments_history || session.mps_adjustments_history.length === 0) {
            return { adjusted: false, description: "Retained (Iteration 1)", badgeClass: "bg-slate-100 text-slate-700 border-slate-300" };
        }
        const lastAdj = session.mps_adjustments_history[session.mps_adjustments_history.length - 1];
        const dmMeta = lastAdj?.[dm];
        if (dmMeta && dmMeta.adjusted) {
            const candText = dmMeta.voted_candidate_index !== undefined ? ` towards Cand ${dmMeta.voted_candidate_index + 1}` : "";
            return {
                adjusted: true,
                description: `⚡ Adjusted${candText} (Iter ${session.current_iteration - 1})`,
                badgeClass: "bg-amber-100 text-amber-800 border-amber-300 font-semibold"
            };
        }
        return {
            adjusted: false,
            description: "🟢 Retained (Voted top choice)",
            badgeClass: "bg-emerald-50 text-emerald-700 border-emerald-300 font-medium"
        };
    }

    // Derived objectives array fed directly to the parallel coordinates plot
    let displayedObjectives = $derived.by<number[][]>(() => {
        if (!session) return [];
        if (activeViewMode === "current_mps") {
            return groupDms.map(dm => extractObjectiveArray(currentMpsMap[dm]));
        }
        if (activeViewMode === "original_mps") {
            return groupDms.map(dm => extractObjectiveArray(originalMpsMap[dm]));
        }
        return session.candidates.map(c => extractObjectiveArray(c.objective_values));
    });

    // Tooltip labels for parallel coordinates lines
    let displayedLineLabels = $derived.by<Record<string, string>>(() => {
        if (!session) return {};
        if (activeViewMode === "current_mps") {
            return Object.fromEntries(
                groupDms.map((dm, i) => [
                    i,
                    `<strong>${dm.toUpperCase()}</strong>: Current Preferred Solution`
                ])
            );
        }
        if (activeViewMode === "original_mps") {
            return Object.fromEntries(
                groupDms.map((dm, i) => [
                    i,
                    `<strong>${dm.toUpperCase()}</strong>: Original Preferred Solution (Iteration 1)`
                ])
            );
        }
        return Object.fromEntries(
            session.candidates.map((c, i) => [
                i,
                `<strong>Candidate ${i + 1}</strong> (${c.fairness_criterion})`
            ])
        );
    });

    // Custom line stroke colors for DM preference curves
    let displayedCustomLineColors = $derived.by<string[]>(() => {
        if (activeViewMode === "current_mps" || activeViewMode === "original_mps") {
            return groupDms.map((_, i) => getDmColor(i).stroke);
        }
        return [];
    });

    // Available DMs: automatically derived from active session, or from group info, or default
    let groupDms = $derived<string[]>(
        session?.dm_ids ??
        (data.group?.user_ids?.length
            ? data.group.user_ids.map((_: any, i: number) => `dm${i + 1}`)
            : ["dm1", "dm2", "dm3", "dm4"])
    );

    let isDecisionPhase = $derived(
        session?.phase === "decision"
    );

    let currentProblem = $derived(
        data.problems?.find((p: any) => Number(p.id) === Number(selectedProblemId))
    );

    let objectiveNameMap = $derived<Record<string, string>>(
        Object.fromEntries(
            (currentProblem?.objectives ?? []).map((o: any) => [o.symbol, o.name])
        )
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

        const probName = currentProblem?.name?.toLowerCase() ?? "";

        if (probId === 1 || probName.includes("river")) {
            // Problem 1: Discrete River Pollution (4 objectives - all maximized)
            // Pareto optimal solutions from datasets/river_poll_4_objs.csv
            const defaultMps: Record<string, Record<string, number>> = {
                "dm1": {"f1": 5.9066, "f2": 3.2894, "f3": 6.5792, "f4": -4.5460},
                "dm2": {"f1": 5.4290, "f2": 3.0121, "f3": 7.2395, "f4": -0.9135},
                "dm3": {"f1": 6.1623, "f2": 2.8839, "f3": 5.2575, "f4": -0.0045},
                "dm4": {"f1": 5.8790, "f2": 3.3617, "f3": 6.6493, "f4": -6.6835},
            };
            active_dms.forEach((dm, i) => {
                mps_payload[dm] = defaultMps[dm] || {
                    "f1": 5.80 + (i * 0.1),
                    "f2": 3.10 + (i * 0.05),
                    "f3": 6.00 + (i * 0.2),
                    "f4": -3.00 + (i * 0.5)
                };
            });
        } else if (probId === 3 || probName.includes("forest") || probName.includes("dmitry")) {
            // Problem 3: Dmitry Forest Problem (Discrete) (4 objectives - all maximized)
            // Pareto optimal solutions from datasets/dmitry_forest_problem_non_dom_solns.csv
            const defaultMps: Record<string, Record<string, number>> = {
                "dm1": {"Rev": 249.5904, "HA": 12497.6850, "Carb": 2880.2038, "DW": 96.9443},
                "dm2": {"Rev": 141.1089, "HA": 20224.8348, "Carb": 3952.1429, "DW": 211.6469},
                "dm3": {"Rev": 86.2129, "HA": 18288.0717, "Carb": 4448.7892, "DW": 206.2755},
                "dm4": {"Rev": 232.3387, "HA": 18328.0238, "Carb": 3347.8052, "DW": 186.4134},
            };
            active_dms.forEach((dm, i) => {
                mps_payload[dm] = defaultMps[dm] || {
                    "Rev": 150.0 + (i * 10),
                    "HA": 18000.0 + (i * 500),
                    "Carb": 3900.0 + (i * 100),
                    "DW": 200.0 + (i * 5)
                };
            });
        } else if (probId === 4 || probName.includes("metall")) {
            // Problem 4: Metallurgical Application (Discrete) (5 objectives)
            // Pareto optimal solutions from metallappl_mop2_m5.npz in metallfronts.zip
            const defaultMps: Record<string, Record<string, number>> = {
                "dm1": {"YS": 796.1600, "UTS": 870.8688, "ELON": 19.5884, "CE": 0.3452, "COST": 6.4521},
                "dm2": {"YS": 598.4819, "UTS": 1898.2363, "ELON": 22.8920, "CE": 1.2768, "COST": 180.4054},
                "dm3": {"YS": 557.6385, "UTS": 576.9771, "ELON": 41.1000, "CE": 0.3058, "COST": 3.3261},
                "dm4": {"YS": 638.7440, "UTS": 612.2479, "ELON": 29.9485, "CE": 0.1782, "COST": 0.9220},
                "dm5": {"YS": 652.8835, "UTS": 1061.3929, "ELON": 30.7727, "CE": 0.5317, "COST": 7.1080},
            };
            active_dms.forEach((dm, i) => {
                mps_payload[dm] = defaultMps[dm] || {
                    "YS": 650.0 + (i * 20),
                    "UTS": 1000.0 + (i * 50),
                    "ELON": 30.0 + (i * 2),
                    "CE": 0.5 + (i * 0.1),
                    "COST": 20.0 + (i * 5)
                };
            });
        } else if (probId === 5 || probName.includes("re34") || probName.includes("crash")) {
            // Problem 5: RE34 Vehicle Crashworthiness (3 objectives)
            // Pareto optimal solutions solved via PyomoIpoptSolver
            const defaultMps: Record<string, Record<string, number>> = {
                "dm1": {"f_1": 1666.4106, "f_2": 6.9593, "f_3": 0.0923},
                "dm2": {"f_1": 1675.4896, "f_2": 6.1428, "f_3": 0.2640},
                "dm3": {"f_1": 1674.3033, "f_2": 9.0811, "f_3": 0.0523},
            };
            active_dms.forEach((dm, i) => {
                mps_payload[dm] = defaultMps[dm] || {
                    "f_1": 1665.0 + (i * 3),
                    "f_2": 7.0 + (i * 0.5),
                    "f_3": 0.1 + (i * 0.03)
                };
            });
        } else {
            // Problem 2: DTLZ2 (3 objectives)
            const defaultMps: Record<string, Record<string, number>> = {
                "dm1": {"f_1": 0.6666, "f_2": 0.6666, "f_3": 0.3333},
                "dm2": {"f_1": 0.6666, "f_2": 0.3333, "f_3": 0.6666},
                "dm3": {"f_1": 0.3333, "f_2": 0.6666, "f_3": 0.6666},
                "dm4": {"f_1": 0.5774, "f_2": 0.5774, "f_3": 0.5774},
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
            max_iterations: 5,
            num_initial_reference_points: 1000,
            most_preferred_solutions: mps_payload
        });

        if (newState) {
            session = newState;
            analystViewMode = "candidates";
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
                <!-- Analyst View Switcher & DM Legend (Analyst Mode Only) -->
                {#if currentRole === "analyst"}
                    <div class="bg-white border rounded-lg p-3 shadow-sm flex flex-wrap items-center justify-between gap-3">
                        <div class="flex items-center gap-2">
                            <span class="text-xs font-bold uppercase tracking-wider text-gray-500">Analyst View:</span>
                            <div class="inline-flex rounded-md shadow-sm" role="group">
                                <button
                                    type="button"
                                    class="px-3 py-1.5 text-xs font-semibold rounded-l-lg border transition-colors {analystViewMode === 'candidates' ? 'bg-blue-600 text-white border-blue-600 shadow' : 'bg-white text-gray-700 hover:bg-gray-100 border-gray-300'}"
                                    onclick={() => analystViewMode = 'candidates'}
                                >
                                    📊 Current Candidates ({session.candidates.length})
                                </button>
                                <button
                                    type="button"
                                    class="px-3 py-1.5 text-xs font-semibold border-t border-b border-r transition-colors {analystViewMode === 'current_mps' ? 'bg-purple-600 text-white border-purple-600 shadow' : 'bg-white text-gray-700 hover:bg-gray-100 border-gray-300'}"
                                    onclick={() => analystViewMode = 'current_mps'}
                                >
                                    🎯 Current DM Preferences ({groupDms.length})
                                </button>
                                <button
                                    type="button"
                                    class="px-3 py-1.5 text-xs font-semibold rounded-r-lg border-t border-b border-r transition-colors {analystViewMode === 'original_mps' ? 'bg-slate-700 text-white border-slate-700 shadow' : 'bg-white text-gray-700 hover:bg-gray-100 border-gray-300'}"
                                    onclick={() => analystViewMode = 'original_mps'}
                                >
                                    📍 Original Baseline ({groupDms.length})
                                </button>
                            </div>
                        </div>

                        {#if analystViewMode === 'current_mps' || analystViewMode === 'original_mps'}
                            <div class="flex items-center gap-2 flex-wrap text-xs">
                                <span class="font-bold text-gray-500">DMs:</span>
                                {#each groupDms as dm, i}
                                    {@const color = getDmColor(i)}
                                    <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full border font-semibold {color.pill}">
                                        <span class="w-2 h-2 rounded-full inline-block shadow-sm" style="background-color: {color.stroke};"></span>
                                        {dm.toUpperCase()}
                                    </span>
                                {/each}
                            </div>
                        {/if}
                    </div>
                {/if}

                <div class="h-64 border rounded shadow-sm bg-white p-2">
                    <VisualizationsPanel
                        problem={currentProblem}
                        solutionsObjectiveValues={displayedObjectives}
                        previousPreferenceType="reference_point"
                        currentPreferenceType="reference_point"
                        lineLabels={displayedLineLabels}
                        customLineColors={displayedCustomLineColors}
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

                {#if session.tie_state?.strategy === "tie_breaker_avgproj"}
                    {@const tiedIndices = session.tie_state?.tied_candidate_indices ?? []}
                    {@const tiedListStr = tiedIndices.map((idx: number) => `Candidate ${idx + 1}`).join(" and ")}
                    <div class="bg-blue-50 text-blue-900 p-4 rounded border border-blue-300 font-medium text-center shadow-sm">
                        <span class="font-bold text-base">🤝 Adjacent Tie Detected between {tiedListStr}!</span>
                        <p class="text-sm mt-1">
                            {session.status === "completed"
                                ? "A compromise solution was generated via Average Projection and chosen as the group consensus."
                                : "A compromise solution was generated via Average Projection onto the Pareto front. Proceed to the next iteration to zoom around this compromise."}
                        </p>
                    </div>
                {/if}

                {#if session.candidates.some((c: any) => c.fairness_criterion?.startsWith("winner_and_"))}
                    <div class="bg-indigo-50/80 border border-indigo-200 text-indigo-900 p-3.5 rounded-lg text-sm shadow-sm flex items-center gap-3">
                        <span class="text-xl">ℹ️</span>
                        <div>
                            <span class="font-bold">Duplicate Candidate Merged:</span> The previous winning solution was identical to this iteration's group-fair solution in both objective and decision spaces. They have been combined into <strong>Candidate 1 (Fair & Last Voted)</strong>, and an additional diversity candidate was generated so the group always evaluates 5 distinct alternatives.
                        </div>
                    </div>
                {/if}

                <!-- Sub-plot Cards Area: Candidates OR DM Preferences -->
                {#if activeViewMode === "candidates"}
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
                                {objectiveNameMap}
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
                {:else if activeViewMode === "current_mps"}
                    <div class="flex flex-col gap-3">
                        <div class="flex items-center justify-between">
                            <h4 class="font-bold text-sm text-purple-950 flex items-center gap-2">
                                🎯 Current Decision Maker Preferred Solutions (Iteration {session.current_iteration})
                            </h4>
                            <span class="text-xs text-purple-700 font-semibold bg-purple-50 px-2.5 py-1 rounded border border-purple-200">
                                Dynamically Adapted Preferences
                            </span>
                        </div>
                        <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4">
                            {#each groupDms as dm, i}
                                {@const color = getDmColor(i)}
                                {@const adjInfo = getDmAdjustmentInfo(dm)}
                                {@const mpsObj = currentMpsMap[dm] ?? {}}
                                <div class="p-4 rounded-lg border-2 shadow-sm flex flex-col justify-between transition-all bg-white" style="border-color: {color.stroke};">
                                    <div>
                                        <div class="flex items-center justify-between mb-2">
                                            <span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-bold border {color.pill}">
                                                <span class="w-2.5 h-2.5 rounded-full inline-block shadow-sm" style="background-color: {color.stroke};"></span>
                                                {dm.toUpperCase()} Preferred
                                            </span>
                                            <span class="text-[11px] px-2 py-0.5 rounded-full border {adjInfo.badgeClass}">
                                                {adjInfo.description}
                                            </span>
                                        </div>
                                        <div class="text-xs space-y-1.5 mt-3">
                                            {#each (currentProblem?.objectives ?? []) as obj}
                                                <div class="flex justify-between border-b pb-1 last:border-0">
                                                    <span class="text-gray-500 font-medium">{obj.name || obj.symbol}:</span>
                                                    <span class="font-mono font-bold text-gray-800">
                                                        {mpsObj[obj.symbol] !== undefined ? Number(mpsObj[obj.symbol]).toFixed(4) : "—"}
                                                    </span>
                                                </div>
                                            {/each}
                                        </div>
                                    </div>
                                    <div class="mt-3 pt-2 border-t text-[11px] text-gray-400 italic text-right">
                                        Active in group fairness ranking
                                    </div>
                                </div>
                            {/each}
                        </div>
                    </div>
                {:else if activeViewMode === "original_mps"}
                    <div class="flex flex-col gap-3">
                        <div class="flex items-center justify-between">
                            <h4 class="font-bold text-sm text-slate-800 flex items-center gap-2">
                                📍 Original Decision Maker Preferred Solutions (Iteration 1 Baseline)
                            </h4>
                            <span class="text-xs text-slate-700 font-semibold bg-slate-100 px-2.5 py-1 rounded border border-slate-300">
                                Baseline Preferences
                            </span>
                        </div>
                        <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4">
                            {#each groupDms as dm, i}
                                {@const color = getDmColor(i)}
                                {@const mpsObj = originalMpsMap[dm] ?? {}}
                                <div class="p-4 rounded-lg border shadow-sm flex flex-col justify-between bg-white border-slate-200">
                                    <div>
                                        <div class="flex items-center justify-between mb-2">
                                            <span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-bold border {color.pill}">
                                                <span class="w-2.5 h-2.5 rounded-full inline-block shadow-sm" style="background-color: {color.stroke};"></span>
                                                {dm.toUpperCase()} Baseline
                                            </span>
                                            <span class="text-[11px] px-2 py-0.5 rounded-full border bg-slate-100 text-slate-700 border-slate-300 font-medium">
                                                Iter 1
                                            </span>
                                        </div>
                                        <div class="text-xs space-y-1.5 mt-3">
                                            {#each (currentProblem?.objectives ?? []) as obj}
                                                <div class="flex justify-between border-b pb-1 last:border-0">
                                                    <span class="text-gray-500 font-medium">{obj.name || obj.symbol}:</span>
                                                    <span class="font-mono font-bold text-gray-800">
                                                        {mpsObj[obj.symbol] !== undefined ? Number(mpsObj[obj.symbol]).toFixed(4) : "—"}
                                                    </span>
                                                </div>
                                            {/each}
                                        </div>
                                    </div>
                                    <div class="mt-3 pt-2 border-t text-[11px] text-gray-400 italic text-right">
                                        Starting aspiration anchor
                                    </div>
                                </div>
                            {/each}
                        </div>
                    </div>
                {/if}
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
                                <td class="p-2.5 font-medium text-gray-700">{objectiveNameMap[key] || key}</td>
                                <td class="p-2.5 font-mono text-gray-900 font-semibold">{Number(val).toFixed(4)}</td>
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        {:else if session && activeViewMode === "candidates"}
            <div class="p-4 bg-white rounded h-full overflow-y-auto">
                <h3 class="font-bold mb-3 text-lg">Candidate Data Table</h3>
                <table class="w-full text-sm text-left border-collapse">
                    <thead>
                        <tr class="border-b-2 border-gray-300">
                            <th class="pb-2 pr-4 font-bold text-gray-700">Candidate</th>
                            <th class="pb-2 pr-4 font-bold text-gray-700">Fairness</th>
                            {#each (currentProblem?.objectives ?? []) as obj}
                                <th class="pb-2 pr-4 font-bold text-gray-700">{obj.name || obj.symbol}</th>
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
                                {#each (currentProblem?.objectives ?? []) as obj}
                                    <td class="py-3 pr-4 font-mono">
                                        {cand.objective_values[obj.symbol] !== undefined ? Number(cand.objective_values[obj.symbol]).toFixed(4) : "—"}
                                    </td>
                                {/each}
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        {:else if session && activeViewMode === "current_mps"}
            <div class="p-4 bg-white rounded h-full overflow-y-auto">
                <div class="flex items-center justify-between mb-3 border-b pb-2">
                    <h3 class="font-bold text-lg text-purple-950 flex items-center gap-2">
                        🎯 Current DM Preferred Solutions Table
                    </h3>
                    <span class="text-xs text-purple-700 font-semibold bg-purple-50 px-2 py-0.5 rounded border border-purple-200">
                        Active In Iteration {session.current_iteration}
                    </span>
                </div>
                <table class="w-full text-sm text-left border-collapse">
                    <thead>
                        <tr class="border-b-2 border-gray-300">
                            <th class="pb-2 pr-4 font-bold text-gray-700">Decision Maker</th>
                            <th class="pb-2 pr-4 font-bold text-gray-700">Status</th>
                            {#each (currentProblem?.objectives ?? []) as obj}
                                <th class="pb-2 pr-4 font-bold text-gray-700">{obj.name || obj.symbol}</th>
                            {/each}
                        </tr>
                    </thead>
                    <tbody>
                        {#each groupDms as dm, i}
                            {@const color = getDmColor(i)}
                            {@const adjInfo = getDmAdjustmentInfo(dm)}
                            {@const mpsObj = currentMpsMap[dm] ?? {}}
                            <tr class="border-b last:border-0 hover:bg-gray-50">
                                <td class="py-3 pr-4 font-semibold">
                                    <span class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-bold border {color.pill}">
                                        <span class="w-2 h-2 rounded-full inline-block shadow-sm" style="background-color: {color.stroke};"></span>
                                        {dm.toUpperCase()}
                                    </span>
                                </td>
                                <td class="py-3 pr-4">
                                    <span class="text-xs px-2 py-0.5 rounded-full border {adjInfo.badgeClass}">
                                        {adjInfo.description}
                                    </span>
                                </td>
                                {#each (currentProblem?.objectives ?? []) as obj}
                                    <td class="py-3 pr-4 font-mono font-medium">
                                        {mpsObj[obj.symbol] !== undefined ? Number(mpsObj[obj.symbol]).toFixed(4) : "—"}
                                    </td>
                                {/each}
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        {:else if session && activeViewMode === "original_mps"}
            <div class="p-4 bg-white rounded h-full overflow-y-auto">
                <div class="flex items-center justify-between mb-3 border-b pb-2">
                    <h3 class="font-bold text-lg text-slate-900 flex items-center gap-2">
                        📍 Original Baseline Preferred Solutions Table
                    </h3>
                    <span class="text-xs text-slate-700 font-semibold bg-slate-100 px-2 py-0.5 rounded border border-slate-300">
                        Immutable Baseline
                    </span>
                </div>
                <table class="w-full text-sm text-left border-collapse">
                    <thead>
                        <tr class="border-b-2 border-gray-300">
                            <th class="pb-2 pr-4 font-bold text-gray-700">Decision Maker</th>
                            <th class="pb-2 pr-4 font-bold text-gray-700">Type</th>
                            {#each (currentProblem?.objectives ?? []) as obj}
                                <th class="pb-2 pr-4 font-bold text-gray-700">{obj.name || obj.symbol}</th>
                            {/each}
                        </tr>
                    </thead>
                    <tbody>
                        {#each groupDms as dm, i}
                            {@const color = getDmColor(i)}
                            {@const mpsObj = originalMpsMap[dm] ?? {}}
                            <tr class="border-b last:border-0 hover:bg-gray-50">
                                <td class="py-3 pr-4 font-semibold">
                                    <span class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-bold border {color.pill}">
                                        <span class="w-2 h-2 rounded-full inline-block shadow-sm" style="background-color: {color.stroke};"></span>
                                        {dm.toUpperCase()}
                                    </span>
                                </td>
                                <td class="py-3 pr-4">
                                    <span class="text-xs px-2 py-0.5 rounded-full border bg-slate-100 text-slate-700 border-slate-300 font-medium">
                                        Iteration 1 Baseline
                                    </span>
                                </td>
                                {#each (currentProblem?.objectives ?? []) as obj}
                                    <td class="py-3 pr-4 font-mono font-medium">
                                        {mpsObj[obj.symbol] !== undefined ? Number(mpsObj[obj.symbol]).toFixed(4) : "—"}
                                    </td>
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
