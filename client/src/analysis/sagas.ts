import { call, fork, put, select, takeEvery } from 'redux-saga/effects';
import * as uuid from 'uuid/v4';
import { assertNotReached } from '../helpers';
import { cancelJob, startJob } from '../job/api';
import { JobState } from '../job/types';
import { AnalysisDetails, AnalysisTypes, DatasetState } from '../messages';
import { RootReducer } from '../store';
import * as analysisActions from './actions';
import { AnalysisState, JobKind } from './types';


// TODO: flip this around - create classes for each analysis type
// classes should provide:
//  + methods for default parameters
//  + creation of a job from current parameters
function getAnalysisDetails(analysisType: AnalysisTypes, dataset: DatasetState): AnalysisDetails {
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
    }
    return assertNotReached("unhandeled analysis type");
}

function selectDataset(state: RootReducer, dataset: string) {
    return state.datasets.byId[dataset];
}

export function* createAnalysisSaga(action: ReturnType<typeof analysisActions.Actions.create>) {
    try {
        const datasetState: DatasetState = yield select(selectDataset, action.payload.dataset)
        const analysis: AnalysisState = {
            id: uuid(),
            dataset: action.payload.dataset,
            resultDetails: getAnalysisDetails(action.payload.analysisType, datasetState),
            frameDetails: { type: AnalysisTypes.SUM_FRAMES, parameters: {} },
            jobs: {},
        }
        yield put(analysisActions.Actions.created(analysis))
        yield put(analysisActions.Actions.run(analysis.id, "FRAME"));
    } catch (e) {
        const timestamp = Date.now();
        const id = uuid();
        yield put(analysisActions.Actions.error(`Error creating analysis: ${e.toString()}`, timestamp, id));
    }
}

function selectAnalysis(state: RootReducer, id: string) {
    return state.analyses.byId[id];
}

function selectJob(state: RootReducer, id: string) {
    return state.jobs.byId[id];
}

export function* cancelOldJob(analysis: AnalysisState, kind: JobKind) {
    if (analysis.jobs[kind] === undefined) {
        return;
    }
    const job: JobState = yield select(selectJob, analysis.jobs[kind]);
    if (job.running !== "DONE") {
        yield call(cancelJob, analysis.jobs[kind]);
    }
}

export function* runAnalysis(analysis: AnalysisState, kind: JobKind) {
    try {
        yield call(cancelOldJob, analysis, kind);

        const jobId = uuid();
        let job;

        // TODO: make it more generic
        if (kind === "RESULT") {
            job = yield call(startJob, jobId, analysis.dataset, analysis.resultDetails);
        } else {
            job = yield call(startJob, jobId, analysis.dataset, analysis.frameDetails);
        }

        return yield put(analysisActions.Actions.running(
            analysis.id,
            job.job,
            kind,
        ));
    } catch (e) {
        const timestamp = Date.now();
        const id = uuid();
        yield put(analysisActions.Actions.error(`Error running analysis: ${e.toString()}`, timestamp, id));
    }
}

export function* runAnalysisSaga(action: ReturnType<typeof analysisActions.Actions.run>) {
    const analysis: AnalysisState = yield select(selectAnalysis, action.payload.id)
    yield fork(runAnalysis, analysis, action.payload.kind);
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
    yield takeEvery(analysisActions.ActionTypes.RUN, runAnalysisSaga);
    yield takeEvery(analysisActions.ActionTypes.SET_FRAMEVIEW_MODE, updateFrameViewMode);
    yield takeEvery(analysisActions.ActionTypes.UPDATE_PARAMETERS, updateFrameViewParams);
}