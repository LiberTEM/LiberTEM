import { buffers, TakeableChannel, Task } from 'redux-saga';
import { actionChannel, call, cancel, fork, put, select, take, takeEvery } from 'redux-saga/effects';
import { v4 as uuid } from 'uuid';
import * as analysisActions from '../analysis/actions';
import { AnalysisState } from '../analysis/types';
import * as channelActions from '../channel/actions';
import * as jobActions from '../job/actions';
import { cancelJob, startJob } from '../job/api';
import { JobRunning, JobState } from '../job/types';
import { AnalysisDetails, DatasetState, DatasetStatus } from '../messages';
import { RootReducer } from '../store';
import * as compoundAnalysisActions from './actions';
import { createOrUpdateAnalysis, createOrUpdateCompoundAnalysis, removeAnalysis, removeCompoundAnalysis } from "./api";
import { CompoundAnalysis, CompoundAnalysisState } from './types';

const selectDataset = (state: RootReducer, dataset: string) => state.datasets.byId[dataset]
const selectCompoundAnalysis = (state: RootReducer, id: string) => state.compoundAnalyses.byId[id]
const selectAnalysis = (state: RootReducer, id: string) => state.analyses.byId[id]
const selectJob = (state: RootReducer, id: string) => state.jobs.byId[id]

export function* cleanupOnRemove(compoundAnalysis: CompoundAnalysis, sidecarTask: Task) {
    while (true) {
        const removeAction = (yield take(compoundAnalysisActions.ActionTypes.REMOVE)) as ReturnType<typeof compoundAnalysisActions.Actions.remove>;
        if (removeAction.payload.id === compoundAnalysis.compoundAnalysis) {
            yield cancel(sidecarTask);
        }
    }
}

export function* createCompoundAnalysisSaga(action: ReturnType<typeof compoundAnalysisActions.Actions.create>) {
    try {
        const datasetState = (yield select(selectDataset, action.payload.dataset)) as DatasetState;
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

        const sidecarTask = (yield fork(analysisSidecar, compoundAnalysis.compoundAnalysis /* , { doAutoStart: true } */)) as Task;

        yield put(compoundAnalysisActions.Actions.created(compoundAnalysis, true));
        yield fork(cleanupOnRemove, compoundAnalysis, sidecarTask);
    } catch (e) {
        const timestamp = Date.now();
        const id = uuid();
        yield put(compoundAnalysisActions.Actions.error(`Error creating analysis: ${(e as Error).toString()}`, timestamp, id));
    }
}

export function* createFromServerState(action: ReturnType<typeof channelActions.Actions.initialState>) {
    for (const msgPart of action.payload.compoundAnalyses) {
        const compoundAnalysis = (yield select(selectCompoundAnalysis, msgPart.compoundAnalysis)) as CompoundAnalysisState;
        const sidecarTask = (yield fork(analysisSidecar, compoundAnalysis.compoundAnalysis /* , { doAutoStart: false } */)) as Task;
        yield fork(cleanupOnRemove, compoundAnalysis, sidecarTask);
    }
}

export function* cancelOldJob(analysis: AnalysisState, jobIndex: number) {
    const jobId = analysis.jobs[jobIndex];
    if (jobId === undefined) {
        return;
    } else {
        const job = (yield select(selectJob, jobId)) as JobState;
        if (job.running !==  JobRunning.DONE) {
            yield call(cancelJob, jobId);
        }
    }
}

export function* createOrUpdate(
    compoundAnalysis: CompoundAnalysisState, analysisId: string | undefined,
    analysisIndex: number, details: AnalysisDetails
): Generator<unknown, string, any> {
    if (analysisId) {
        // update the analysis on the server:
        yield call(createOrUpdateAnalysis,
            compoundAnalysis.compoundAnalysis, analysisId,
            compoundAnalysis.dataset, details
        );
        yield put(analysisActions.Actions.updated(analysisId, details));

        const analysis = (yield select(selectAnalysis, analysisId)) as AnalysisState;
        const jobs = analysis.jobs ? analysis.jobs : [];

        for (const oldJobId of jobs) {
            const job = (yield select(selectJob, oldJobId)) as JobState;
            if (job && job.running !== JobRunning.DONE) {
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

        const updatedCompoundAnalysis = (yield select(selectCompoundAnalysis, compoundAnalysis.compoundAnalysis)) as CompoundAnalysisState;

        yield call(
            createOrUpdateCompoundAnalysis,
            updatedCompoundAnalysis.compoundAnalysis,
            updatedCompoundAnalysis.dataset,
            updatedCompoundAnalysis.details,
        );
        return newAnalysisId;
    }
}

export function* updateParams(compoundAnalysisId: string, analysisIndex: number, details: AnalysisDetails) {
    const compoundAnalysis = (yield select(selectCompoundAnalysis, compoundAnalysisId)) as CompoundAnalysisState;
    const existingAnalysisId = compoundAnalysis.details.analyses[analysisIndex];
    return (yield call(createOrUpdate, compoundAnalysis, existingAnalysisId, analysisIndex, details)) as string;
}

export function* setParamsSaga(action: compoundAnalysisActions.ActionParts["setParams"]) {
    const { compoundAnalysis, analysisIndex, details } = action.payload;
    yield call(updateParams, compoundAnalysis.compoundAnalysis, analysisIndex, details);
}

export function* paramsSyncSidecar() {
    const runOrParamsChannel = (yield actionChannel(compoundAnalysisActions.ActionTypes.SET_PARAMS, buffers.sliding(2))) as TakeableChannel<compoundAnalysisActions.ActionTypes.SET_PARAMS>;

    while (true) {
        const action = (yield take(runOrParamsChannel)) as compoundAnalysisActions.ActionParts["setParams"];

        yield call(setParamsSaga, action);
    }
}

export function* analysisSidecar(compoundAnalysisId: string /* , options: { doAutoStart: boolean } */) {
    // channel for incoming actions:
    // all actions that arrive while we block in `call` will be buffered here.
    // because the buffer is sliding of size 2, we only keep the latest two actions!
    const runOrParamsChannel = (yield actionChannel(compoundAnalysisActions.ActionTypes.RUN, buffers.sliding(2))) as TakeableChannel<
        compoundAnalysisActions.ActionTypes.RUN
    >;

    while (true) {
        try {
            const action = (yield take(runOrParamsChannel)) as compoundAnalysisActions.ActionParts["run"];

            // ignore actions meant for other analyses
            if (action.payload.id !== compoundAnalysisId) {
                continue;
            }

            // get the current state incl. configuration
            const compoundAnalysis = (yield select(selectCompoundAnalysis, compoundAnalysisId)) as CompoundAnalysisState;
            const { analysisIndex, details } = action.payload;

            // update parameters on the server and in redux:
            const analysisId = (yield call(updateParams, compoundAnalysis.compoundAnalysis, analysisIndex, details)) as string;

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
            yield put(compoundAnalysisActions.Actions.error(`Error running analysis: ${(e as Error).toString()}`, timestamp, id));
        }
    }
}

function* removeJobsForAnalysis(analysis: AnalysisState) {
    for (const oldJobId of analysis.jobs) {
        const job = (yield select(selectJob, oldJobId)) as JobState;
        if (job && job.running !== JobRunning.DONE) {
            // wait until the job is cancelled:
            yield call(cancelJob, oldJobId);
        }
    }
}

export function* doRemoveAnalysisSaga(action: ReturnType<typeof compoundAnalysisActions.Actions.remove>) {
    const compoundAnalysis = (yield select(selectCompoundAnalysis, action.payload.id)) as CompoundAnalysisState;
    try {
        for (const analysisId of compoundAnalysis.details.analyses) {
            const analysis = (yield select(selectAnalysis, analysisId)) as AnalysisState;
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

    (yield fork(paramsSyncSidecar)) as Task;
}
