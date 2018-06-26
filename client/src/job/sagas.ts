// TODO: here we could have a saga that listenes to events like
// fromAnalysis.ActionType.CREATED and UPDATE_PARAM and starts jobs accordingly
// needs to be debounced to not create unnecessary jobs, at the beginning the
// starting of a new job should even wait for completion of the current job
// (need to make sure to eventually start a job for the current parameters,
// so the user interface and results dont get out of sync)
// 
// later: canceling: when new parameters for an analysis come in, cancel the old
// job and kick off a new one