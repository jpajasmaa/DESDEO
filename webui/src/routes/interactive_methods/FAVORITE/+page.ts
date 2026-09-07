import type { PageLoad } from './$types';
import {
  getProblemsInfoProblemAllInfoGet,
  getGroupInfoGdmGetGroupInfoPost
} from '$lib/gen/endpoints/DESDEOFastAPI';
import type { ProblemInfo, GroupPublic } from '$lib/gen/endpoints/DESDEOFastAPI';

export const load: PageLoad = async ({ url }) => {
  let problems: ProblemInfo[] = [];
  let group: GroupPublic | null = null;

  try {
    const res = await getProblemsInfoProblemAllInfoGet();
    if (res.status === 200) {
      problems = res.data as ProblemInfo[];
    }
  } catch (err) {
    console.error("Error connecting to DESDEO API:", err);
  }

  const groupId = url.searchParams.get('group');
  if (groupId) {
    try {
      const gRes = await getGroupInfoGdmGetGroupInfoPost({ group_id: parseInt(groupId) });
      if (gRes.status === 200) {
        group = gRes.data as GroupPublic;
      }
    } catch (err) {
      console.warn("Could not fetch group info for groupId:", groupId, err);
    }
  } else {
    // Try fetching the default group (group_id: 1) created by db_init_gdm
    try {
      const gRes = await getGroupInfoGdmGetGroupInfoPost({ group_id: 1 });
      if (gRes.status === 200) {
        group = gRes.data as GroupPublic;
      }
    } catch {
      // Group 1 might not exist or be accessible, fallback handled by UI
    }
  }

  return {
    problems,
    group
  };
};
