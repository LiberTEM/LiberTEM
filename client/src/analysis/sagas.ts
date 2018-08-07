import { call, put, select, take, takeEvery } from 'redux-saga/effects';
import * as uuid from 'uuid/v4';
import { assertNotReached } from '../helpers';
import { cancelJob, startJob } from '../job/api';
import { JobState } from '../job/types';
import { AnalysisDetails, AnalysisTypes, DatasetState } from '../messages';
import { RootReducer } from '../store';
import * as analysisActions from './actions';
import { AnalysisState } from './types';


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
    }
    return assertNotReached("unhandeled analysis type");
}

function selectDataset(state: RootReducer, dataset: string) {
    return state.dataset.byId[dataset];
}

export function* createAnalysisSaga(action: ReturnType<typeof analysisActions.Actions.create>) {
    try {
        const datasetState: DatasetState = yield select(selectDataset, action.payload.dataset)
        const shape = datasetState.params.shape;
        const width = shape[1];
        const height = shape[0];
        const analysis: AnalysisState = {
            id: uuid(),
            dataset: action.payload.dataset,
            details: getAnalysisDetails(action.payload.analysisType, datasetState),
            preview: { mode: "AVERAGE", pick: { x: width / 2, y: height / 2 } },
            currentJob: "",
        }
        yield put(analysisActions.Actions.created(analysis))
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
    return state.job.byId[id];
}

export function* cancelOldJob(analysis: AnalysisState) {
    if (analysis.currentJob === "") {
        return;
    }
    const job: JobState = yield select(selectJob, analysis.currentJob);
    if (job.running !== "DONE") {
        yield call(cancelJob, analysis.currentJob);
    }
}

export function* runAnalysisSaga(action: ReturnType<typeof analysisActions.Actions.run>) {
    try {
        const analysis: AnalysisState = yield select(selectAnalysis, action.payload.id)

        yield call(cancelOldJob, analysis);

        const jobId = uuid();
        const job = yield call(startJob, jobId, analysis.dataset, analysis.details);
        return yield put(analysisActions.Actions.running(
            action.payload.id,
            job.job,
        ))
    } catch (e) {
        const timestamp = Date.now();
        const id = uuid();
        yield put(analysisActions.Actions.error(`Error running analysis: ${e.toString()}`, timestamp, id));
    }
}

export function* cancelJobOnRemove() {
    while (true) {
        const action: ReturnType<typeof analysisActions.Actions.remove> = yield take(analysisActions.ActionTypes.REMOVE);
        const analysis: AnalysisState = yield select(selectAnalysis, action.payload.id)
        if (analysis && analysis.currentJob) {
            yield call(cancelJob, analysis.currentJob);
        }
    }
}

export function* analysisRootSaga() {
    yield takeEvery(analysisActions.ActionTypes.CREATE, createAnalysisSaga);
    yield takeEvery(analysisActions.ActionTypes.RUN, runAnalysisSaga);
    yield takeEvery(analysisActions.ActionTypes.REMOVE, cancelJobOnRemove);
}