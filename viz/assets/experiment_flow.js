// Batch execution shares the solver with live debugging, but never waits for
// browser playback controls. Keep both configuration entry points consistent.
export async function startBatchExperiment(post, sceneId, navigate) {
  if (!sceneId) throw Error('请先选择或保存完整场景');
  const job = await post('/launch-scene', {id: sceneId});
  navigate('/runs/' + encodeURIComponent(job.id) + '/progress?next=replay');
  return job;
}

export function completedDestination(job, search) {
  const query = new URLSearchParams(search);
  if (query.get('next') === 'scene' && job.physical_model_id) {
    const model = encodeURIComponent(query.get('model') || job.physical_model_id);
    const draft = query.get('scene_draft');
    return '/scene?model=' + model + (draft ? '&scene_draft=' + encodeURIComponent(draft) : '');
  }
  if (query.get('next') === 'replay' || job.automatic) {
    return '/results?run=' + encodeURIComponent(job.id) + '&view=scene';
  }
  return (job.native_preset || job.native_scene) ? job.result_url :
    job.preview_url || job.runtime_url || job.result_url;
}
