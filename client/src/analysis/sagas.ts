import { buffers } from 'redux-saga';
import { actionChannel, call, cancel, fork, put, select, take, takeEvery } from 'redux-saga/effects';
import uuid from 'uuid/v4';
import * as jobActions from '../job/actions';
import { cancelJob, startJob } from '../job/api';
import { JobState } from '../job/types';
import { DatasetState, DatasetStatus } from '../messages';
import { RootReducer } from '../store';
import * as analysisActions from './actions';
import { AnalysisState } from './types';

function selectDataset(state: RootReducer, dataset: string) {
    return state.datasets.byId[dataset];
}

function selectAnalysis(state: RootReducer, id: string) {
    return state.analyses.byId[id];
}

function selectJob(state: RootReducer, id: string) {
    return state.jobs.byId[id];
}


export function* createAnalysisSaga(action: ReturnType<typeof analysisActions.Actions.create>) {
    try {
        const datasetState: DatasetState = yield select(selectDataset, action.payload.dataset)
        if (datasetState.status !== DatasetStatus.OPEN) {
            throw new Error("invalid dataset status");
        }
        const analysis: AnalysisState = {
            id: uuid(),
            dataset: action.payload.dataset,
            mainAnalysisType: action.payload.analysisType,
            jobs: [],
            jobHistory: [],
        }

        const sidecarTask = yield fork(analysisSidecar, analysis.id);

        yield put(analysisActions.Actions.created(analysis));

        while (true) {
            const removeAction: ReturnType<typeof analysisActions.Actions.remove> = yield take(analysisActions.ActionTypes.REMOVE);
            if (removeAction.payload.id === analysis.id) {
                yield cancel(sidecarTask);
            }
        }
    } catch (e) {
        const timestamp = Date.now();
        const id = uuid();
        yield put(analysisActions.Actions.error(`Error creating analysis: ${e.toString()}`, timestamp, id));
    }
}

export function* cancelOldJob(analysis: AnalysisState, jobIndex: number) {
    const jobId = analysis.jobs[jobIndex];
    if (jobId === undefined) {
        return;
    } else {
        const job: JobState = yield select(selectJob, jobId);
        if (job.running !== "DONE") {
            yield call(cancelJob, jobId);
        }
    }
}

export function* analysisSidecar(analysisId: string) {
    // channel for incoming actions:
    // all actions that arrive while we block in `call` will be buffered here.
    // because the buffer is sliding of size 1, we only keep the latest action!
    const runOrParamsChannel = yield actionChannel(analysisActions.ActionTypes.RUN, buffers.sliding(2));

    while (true) {
        try {
            const action: analysisActions.ActionParts["run"] = yield take(runOrParamsChannel);

            // ignore actions meant for other analyses
            if (action.payload.id !== analysisId) {
                continue;
            }

            // get the current state incl. configuration
            const analysis: AnalysisState = yield select(selectAnalysis, analysisId);
            const { jobIndex, parameters } = action.payload;

            // prepare running the job:
            const jobId = uuid();
            yield put(jobActions.Actions.create(jobId, analysis.dataset, Date.now()));
            yield put(analysisActions.Actions.prepareRun(analysis.id, jobIndex, jobId));

            const oldJobId = analysis.jobs[jobIndex];
            if (oldJobId !== undefined) {
                const job: JobState = yield select(selectJob, oldJobId);
                if (job && job.running !== "DONE") {
                    // wait until the job is cancelled:
                    yield call(cancelJob, oldJobId);
                }
            }

            // FIXME: we have a race here, as the websocket msg FINISH_JOB may
            // arrive before call(startJob, ...) returns. this causes the apply button
            // to feel unresponsive (the action gets done, but only after we finish here...)
            // best reproduced in "Slow 3G" network simulation mode in devtools

            // wait until the job is started
            yield call(startJob, jobId, analysis.dataset, parameters);
            yield put(analysisActions.Actions.running(analysis.id, jobId, jobIndex));
        } catch (e) {
            const timestamp = Date.now();
            const id = uuid();
            yield put(analysisActions.Actions.error(`Error running analysis: ${e.toString()}`, timestamp, id));
        }
    }
}

export function* doRemoveAnalysisSaga(action: ReturnType<typeof analysisActions.Actions.remove>) {
    // const analysis: AnalysisState = yield select(selectAnalysis, action.payload.id)
    try {
        // TODO: cancel all jobs! loop over all of them...
        // yield call(cancelOldJob, analysis, "RESULT");
        // yield call(cancelOldJob, analysis, "FRAME");
    } finally {
        yield put(analysisActions.Actions.removed(action.payload.id));
    }
}

export function* analysisRootSaga() {
    yield takeEvery(analysisActions.ActionTypes.CREATE, createAnalysisSaga);
    yield takeEvery(analysisActions.ActionTypes.REMOVE, doRemoveAnalysisSaga);
}