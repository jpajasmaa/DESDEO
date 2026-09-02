import type {
  FavoriteInitRequest,
  FavoriteVoteRequest,
  FavoriteSessionState,
} from "./types";

const API_BASE = "/api/favorite";

/**
 * Initializes a new Favorite method session.
 */
export async function initializeFavoriteSession(
  request: FavoriteInitRequest
): Promise<FavoriteSessionState | null> {
  try {
    const res = await fetch(`${API_BASE}/init`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    if (!res.ok) {
      console.error("Failed to initialize Favorite session:", res.status, await res.text());
      return null;
    }

    return await res.json();
  } catch (err) {
    console.error("Error initializing Favorite session:", err);
    return null;
  }
}

/**
 * Fetches the current snapshot of an active session (polled by Analyst and DMs).
 */
export async function getFavoriteState(
  sessionId: string
): Promise<FavoriteSessionState | null> {
  try {
    const res = await fetch(`${API_BASE}/state/${sessionId}`);
    if (!res.ok) {
      console.error("Failed to fetch Favorite state:", res.status);
      return null;
    }
    return await res.json();
  } catch (err) {
    console.error("Error fetching Favorite state:", err);
    return null;
  }
}

/**
 * Submits a DM's favorite candidate choice.
 */
export interface VoteResponse {
  message: string;
  is_ready: boolean;
  current_votes: Record<string, number>;
  phase?: "consensus_reaching" | "decision";
  status: "voting" | "revote_pending" | "completed";
  tie_state?: any;
  final_solution?: any;
}

export async function submitFavoriteVote(
  sessionId: string,
  vote: FavoriteVoteRequest
): Promise<VoteResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/vote/${sessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(vote),
    });

    if (!res.ok) {
      console.error("Failed to submit vote:", res.status, await res.text());
      return null;
    }

    return await res.json();
  } catch (err) {
    console.error("Error submitting vote:", err);
    return null;
  }
}

/**
 * Advances the session to the next zooming iteration. Triggered by Analyst.
 */
export async function iterateFavoriteSession(
  sessionId: string
): Promise<FavoriteSessionState | null> {
  try {
    const res = await fetch(`${API_BASE}/iterate/${sessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    if (!res.ok) {
      console.error("Failed to iterate session:", res.status, await res.text());
      return null;
    }

    return await res.json();
  } catch (err) {
    console.error("Error iterating Favorite session:", err);
    return null;
  }
}
