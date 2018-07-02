import { call, put, select, takeEvery } from 'redux-saga/effects';
import * as uuid from 'uuid/v4';
import { assertNotReached } from '../helpers';
import { startJob } from '../job/api';
import { DatasetState } from '../messages';
import { RootReducer } from '../store';
import * as analysisActions from './actions';
import { AnalysisDetails, AnalysisState, AnalysisTypes } from './types';


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
                parameters: {},
            };
        }
    }
    return assertNotReached("unhandeled analysis type");
}

export function* createAnalysisSaga(action: ReturnType<typeof analysisActions.Actions.create>) {
    function selectDataset(state: RootReducer, dataset: string) {
        return state.dataset.byId[dataset];
    }

    const datasetState: DatasetState = yield select(selectDataset, action.payload.dataset)
    const analysis: AnalysisState = {
        id: uuid(),
        dataset: action.payload.dataset,
        details: getAnalysisDetails(action.payload.analysisType, datasetState),
        currentJob: "",
    }
    yield put(analysisActions.Actions.created(analysis))
}

function selectAnalysis(state: RootReducer, id: string) {
    return state.analyses.byId[id];
}

export function* runAnalysisSaga(action: ReturnType<typeof analysisActions.Actions.run>) {
    try {
        const analysis: AnalysisState = yield select(selectAnalysis, action.payload.id)
        const masks = [
            analysis.details.parameters
        ];
        const jobId = uuid();
        const job = yield call(startJob, jobId, analysis.dataset, masks);
        return yield put(analysisActions.Actions.running(
            action.payload.id,
            job.job,
        ))
    } catch (e) {
        yield put(analysisActions.Actions.error(`Error running analysis: ${e.toString()}`));
    }
}

export function* analysisRootSaga() {
    yield takeEvery(analysisActions.ActionTypes.CREATE, createAnalysisSaga);
    yield takeEvery(analysisActions.ActionTypes.RUN, runAnalysisSaga);
}