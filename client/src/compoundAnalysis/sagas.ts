import { buffers, Task } from 'redux-saga';
import { actionChannel, call, cancel, fork, put, select, take, takeEvery } from 'redux-saga/effects';
import uuid from 'uuid/v4';
import * as analysisActions from '../analysis/actions';
import { AnalysisState } from '../analysis/types';
import * as channelActions from '../channel/actions';
import * as jobActions from '../job/actions';
import { cancelJob, startJob } from '../job/api';
import { JobState } from '../job/types';
import { AnalysisDetails, DatasetState, DatasetStatus } from '../messages';
import { RootReducer } from '../store';
import * as compoundAnalysisActions from './actions';
import { createOrUpdateAnalysis, createOrUpdateCompoundAnalysis, removeAnalysis, removeCompoundAnalysis } from "./api";
import { CompoundAnalysis, CompoundAnalysisState } from './types';

function selectDataset(state: RootReducer, dataset: string) {
    return state.datasets.byId[dataset];
}

function selectCompoundAnalysis(state: RootReducer, id: string) {
    return state.compoundAnalyses.byId[id];
}

function selectAnalysis(state: RootReducer, id: string) {
    return state.analyses.byId[id];
}

function selectJob(state: RootReducer, id: string) {
    return state.jobs.byId[id];
}

export function* cleanupOnRemove(compoundAnalysis: CompoundAnalysis, sidecarTask: Task) {
    while (true) {
        const removeAction: ReturnType<typeof compoundAnalysisActions.Actions.remove> = yield take(compoundAnalysisActions.ActionTypes.REMOVE);
        if (removeAction.payload.id === compoundAnalysis.compoundAnalysis) {
            yield cancel(sidecarTask);
        }
    }
}

export function* createCompoundAnalysisSaga(action: ReturnType<typeof compoundAnalysisActions.Actions.create>) {
    try {
        const datasetState: DatasetState = yield select(selectDataset, action.payload.dataset)
        if (datasetState.status !== DatasetStatus.OPEN) {
            throw new Error("invalid dataset status");
        }
        const compoundAnalysis: CompoundAnalysis = {
            compoundAnalysis: uuid(),
            dataset: action.payload.dataset,
            details: {
                mainType: action.payload.analysisType,
                analyses: [],
            }
        }

        yield call(
            createOrUpdateCompoundAnalysis,
            compoundAnalysis.compoundAnalysis,
            compoundAnalysis.dataset,
            compoundAnalysis.details,
        );

        const sidecarTask = yield fork(analysisSidecar, compoundAnalysis.compoundAnalysis, { doAutoStart: true });

        yield put(compoundAnalysisActions.Actions.created(compoundAnalysis, true));
        yield fork(cleanupOnRemove, compoundAnalysis, sidecarTask);
    } catch (e) {
        const timestamp = Date.now();
        const id = uuid();
        yield put(compoundAnalysisActions.Actions.error(`Error creating analysis: ${e.toString()}`, timestamp, id));
    }
}

export function* createFromServerState(action: ReturnType<typeof channelActions.Actions.initialState>) {
    for (const msgPart of action.payload.compoundAnalyses) {
        const compoundAnalysis: CompoundAnalysisState = yield select(selectCompoundAnalysis, msgPart.compoundAnalysis);
        const sidecarTask = yield fork(analysisSidecar, compoundAnalysis.compoundAnalysis, { doAutoStart: false });
        yield fork(cleanupOnRemove, compoundAnalysis, sidecarTask);
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

export function* createOrUpdate(
    compoundAnalysis: CompoundAnalysisState, analysisId: string | undefined,
    analysisIndex: number, details: AnalysisDetails
) {
    if (analysisId) {
        // update the analysis on the server:
        yield call(createOrUpdateAnalysis,
            compoundAnalysis.compoundAnalysis, analysisId,
            compoundAnalysis.dataset, details
        );
        yield put(analysisActions.Actions.updated(analysisId, details));

        const analysis: AnalysisState = yield select(selectAnalysis, analysisId);
        const jobs = analysis.jobs ? analysis.jobs : [];

        for (const oldJobId of jobs) {
            const job: JobState = yield select(selectJob, oldJobId);
            if (job && job.running !== "DONE") {
                // wait until the job is cancelled:
                yield call(cancelJob, oldJobId);
            }
        }
        return analysisId;
    } else {
        // create the analysis on the server:
        const newAnalysisId = uuid();
        yield call(createOrUpdateAnalysis,
            compoundAnalysis.compoundAnalysis, newAnalysisId,
            compoundAnalysis.dataset, details
        );
        yield put(analysisActions.Actions.created({
            id: newAnalysisId,
            dataset: compoundAnalysis.dataset,
            details,
            jobs: [],
        }, compoundAnalysis.compoundAnalysis, analysisIndex));

        const updatedCompoundAnalysis = yield select(selectCompoundAnalysis, compoundAnalysis.compoundAnalysis);

        yield call(
            createOrUpdateCompoundAnalysis,
            updatedCompoundAnalysis.compoundAnalysis,
            updatedCompoundAnalysis.dataset,
            updatedCompoundAnalysis.details,
        );
        return newAnalysisId;
    }
}

export function* analysisSidecar(compoundAnalysisId: string, options: { doAutoStart: boolean }) {
    // channel for incoming actions:
    // all actions that arrive while we block in `call` will be buffered here.
    // because the buffer is sliding of size 1, we only keep the latest action!
    const runOrParamsChannel = yield actionChannel(compoundAnalysisActions.ActionTypes.RUN, buffers.sliding(2));

    while (true) {
        try {
            const action: compoundAnalysisActions.ActionParts["run"] = yield take(runOrParamsChannel);

            // ignore actions meant for other analyses
            if (action.payload.id !== compoundAnalysisId) {
                continue;
            }

            // get the current state incl. configuration
            const compoundAnalysis: CompoundAnalysisState = yield select(selectCompoundAnalysis, compoundAnalysisId);
            const { analysisIndex, details } = action.payload;

            const existingAnalysisId = compoundAnalysis.details.analyses[analysisIndex];
            const analysisId = yield call(createOrUpdate, compoundAnalysis, existingAnalysisId, analysisIndex, details);

            // prepare running the job:
            const jobId = uuid();
            yield put(jobActions.Actions.create(jobId, analysisId, Date.now()));

            // FIXME: we have a race here, as the websocket msg FINISH_JOB may
            // arrive before call(startJob, ...) returns. this causes the apply button
            // to feel unresponsive (the action gets done, but only after we finish here...)
            // best reproduced in "Slow 3G" network simulation mode in devtools

            // wait until the job is started
            yield call(startJob, jobId, analysisId);
            yield put(compoundAnalysisActions.Actions.running(compoundAnalysis.compoundAnalysis, jobId, analysisIndex));
            // tslint:disable-next-line:no-empty
        } catch (e) {
            const timestamp = Date.now();
            const id = uuid();
            yield put(compoundAnalysisActions.Actions.error(`Error running analysis: ${e.toString()}`, timestamp, id));
        }
    }
}

function* removeJobsForAnalysis(analysis: AnalysisState) {
    for (const oldJobId of analysis.jobs) {
        const job: JobState = yield select(selectJob, oldJobId);
        if (job && job.running !== "DONE") {
            // wait until the job is cancelled:
            yield call(cancelJob, oldJobId);
        }
    }
}

export function* doRemoveAnalysisSaga(action: ReturnType<typeof compoundAnalysisActions.Actions.remove>) {
    const compoundAnalysis: CompoundAnalysisState = yield select(selectCompoundAnalysis, action.payload.id);
    try {
        for (const analysisId of compoundAnalysis.details.analyses) {
            const analysis: AnalysisState = yield select(selectAnalysis, analysisId);
            yield call(removeJobsForAnalysis, analysis);
            yield call(removeAnalysis, compoundAnalysis.compoundAnalysis, analysisId);
            yield put(analysisActions.Actions.removed(analysisId));
        }
        yield call(removeCompoundAnalysis, action.payload.id);
    } finally {
        yield put(compoundAnalysisActions.Actions.removed(action.payload.id));
    }
}

export function* analysisRootSaga() {
    yield takeEvery(compoundAnalysisActions.ActionTypes.CREATE, createCompoundAnalysisSaga);
    yield takeEvery(compoundAnalysisActions.ActionTypes.REMOVE, doRemoveAnalysisSaga);
    yield takeEvery(channelActions.ActionTypes.INITIAL_STATE, createFromServerState);
}