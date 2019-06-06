import { buffers } from 'redux-saga';
import { actionChannel, call, cancel, fork, put, select, take, takeEvery } from 'redux-saga/effects';
import uuid from 'uuid/v4';
import { assertNotReached } from '../helpers';
import * as jobActions from '../job/actions';
import { cancelJob, startJob } from '../job/api';
import { JobState } from '../job/types';
import { AnalysisDetails, AnalysisTypes, DatasetOpen, DatasetState, DatasetStatus } from '../messages';
import { RootReducer } from '../store';
import * as analysisActions from './actions';
import { AnalysisState, JobKind } from './types';


// TODO: flip this around - create classes for each analysis type
// classes should provide:
//  + methods for default parameters
//  + creation of a job from current parameters
function getAnalysisDetails(analysisType: AnalysisTypes, dataset: DatasetOpen): AnalysisDetails {
    const shape = dataset.params.shape;
    const width = shape[3];
    const height = shape[2];
    const minLength = Math.min(width, height);

    switch (analysisType) {
        case AnalysisTypes.APPLY_DISK_MASK: {
            return {
                type: analysisType,
                parameters: {
                    shape: "disk",
                    cx: width / 2,
                    cy: height / 2,
                    r: minLength / 2,
                }
            };
        }
        case AnalysisTypes.APPLY_RING_MASK: {
            return {
                type: analysisType,
                parameters: {
                    shape: "ring",
                    cx: width / 2,
                    cy: height / 2,
                    ri: minLength / 4,
                    ro: minLength / 2,
                }
            }
        }
        case AnalysisTypes.CENTER_OF_MASS: {
            return {
                type: analysisType,
                parameters: {
                    shape: "com",
                    cx: width / 2,
                    cy: height / 2,
                    r: minLength / 2,
                },
            };
        }
        case AnalysisTypes.APPLY_POINT_SELECTOR: {
            return {
                type: analysisType,
                parameters: {
                    shape: "point",
                    cx: width / 2,
                    cy: width / 2,
                }
            }
        }
        case AnalysisTypes.SUM_FRAMES: {
            return {
                type: AnalysisTypes.SUM_FRAMES,
                parameters: {},
            }
        }
        case AnalysisTypes.PICK_FRAME: {
            return {
                type: AnalysisTypes.PICK_FRAME,
                parameters: {
                    x: Math.round(width / 2),
                    y: Math.round(height / 2),
                }
            }
        }
        case AnalysisTypes.RADIAL_FOURIER: {
            return {
                type: analysisType,
                parameters: {
                    shape: "radial_fourier",
                    cx: width / 2,
                    cy: height / 2,
                    ri: minLength / 4,
                    ro: minLength / 2,
                    n_bins: 1,
                    max_order: 8
                }
            }
        }
    }
    return assertNotReached("unhandeled analysis type");
}

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
            resultDetails: getAnalysisDetails(action.payload.analysisType, datasetState),
            frameDetails: { type: AnalysisTypes.SUM_FRAMES, parameters: {} },
            jobs: {},
            jobHistory: {
                FRAME: [],
                RESULT: [],
            }
        }

        const sidecarTask = yield fork(analysisSidecar, analysis.id);

        yield put(analysisActions.Actions.created(analysis));
        yield put(analysisActions.Actions.run(analysis.id, "FRAME"));

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

export function* cancelOldJob(analysis: AnalysisState, kind: JobKind) {
    const jobId = analysis.jobs[kind]
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
    const runOrParamsChannel = yield actionChannel(analysisActions.ActionTypes.RUN, buffers.sliding(1));

    while (true) {
        try {
            const action: analysisActions.ActionParts["run"] = yield take(runOrParamsChannel);

            // ignore actions meant for other analyses
            if (action.payload.id !== analysisId) {
                continue;
            }

            // get the current state incl. configuration
            const analysis: AnalysisState = yield select(selectAnalysis, analysisId);
            const { kind } = action.payload;

            // prepare running the job:
            const jobId = uuid();
            yield put(jobActions.Actions.create(jobId, analysis.dataset, Date.now()));
            yield put(analysisActions.Actions.prepareRun(analysis.id, kind, jobId));

            const oldJobId = analysis.jobs[kind];
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
            if (kind === "FRAME") {
                yield call(startJob, jobId, analysis.dataset, analysis.frameDetails);
            } else if (kind === "RESULT") {
                yield call(startJob, jobId, analysis.dataset, analysis.resultDetails);
            }
            yield put(analysisActions.Actions.running(analysis.id, jobId, kind))
        } catch (e) {
            const timestamp = Date.now();
            const id = uuid();
            yield put(analysisActions.Actions.error(`Error running analysis: ${e.toString()}`, timestamp, id));
        }
    }
}

export function* updateFrameViewMode(action: ReturnType<typeof analysisActions.Actions.setFrameViewMode>) {
    yield put(analysisActions.Actions.run(action.payload.id, "FRAME"));
}

export function* updateFrameViewParams(action: ReturnType<typeof analysisActions.Actions.updateParameters>) {
    if (action.payload.kind === "FRAME") {
        yield put(analysisActions.Actions.run(action.payload.id, "FRAME"));
    }
}

export function* doRemoveAnalysisSaga(action: ReturnType<typeof analysisActions.Actions.remove>) {
    const analysis: AnalysisState = yield select(selectAnalysis, action.payload.id)
    try {
        yield call(cancelOldJob, analysis, "RESULT");
        yield call(cancelOldJob, analysis, "FRAME");
    } finally {
        yield put(analysisActions.Actions.removed(action.payload.id));
    }
}

export function* analysisRootSaga() {
    yield takeEvery(analysisActions.ActionTypes.CREATE, createAnalysisSaga);
    yield takeEvery(analysisActions.ActionTypes.REMOVE, doRemoveAnalysisSaga);
    yield takeEvery(analysisActions.ActionTypes.SET_FRAMEVIEW_MODE, updateFrameViewMode);
    yield takeEvery(analysisActions.ActionTypes.UPDATE_PARAMETERS, updateFrameViewParams);
}