export interface FairSolution {
  fairness_criterion: string;
  fairness_value: number;
  objective_values: Record<string, number>;
  objectives?: Record<string, number>;
}

export interface FavoriteInitRequest {
  problem_id?: number | string;
  dm_ids?: string[];
  total_n_of_candidates?: number;
  candidate_generation_options?: string;
  max_iterations?: number;
  num_initial_reference_points?: number;
  most_preferred_solutions?: Record<string, Record<string, number>>;
}

export interface FavoriteVoteRequest {
  dm_id: string;
  vote_idx: number;
}

export interface TieState {
  tied_candidate_indices?: number[];
  resolved_winner_idx?: number;
  [key: string]: any;
}

export interface FavoriteSessionState {
  session_id: string;
  problem_id: number;
  dm_ids: string[];
  current_iteration: number;
  max_iterations: number;
  phase?: "consensus_reaching" | "decision";
  status: "voting" | "revote_pending" | "completed";
  candidates: FairSolution[];
  current_votes: Record<string, number>;
  tie_state?: TieState | null;
  final_solution?: FairSolution | null;
  options?: Record<string, any>;
  results_history?: any[];
}
