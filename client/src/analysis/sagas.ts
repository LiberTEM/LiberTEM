import { call, put, select, takeEvery } from 'redux-saga/effects';
import * as uuid from 'uuid/v4';
import { assertNotReached } from '../helpers';
import { startJob } from '../job/api';
import { RootReducer } from '../store';
import * as analysisActions from './actions';
import { Analysis, AnalysisDetails, AnalysisTypes } from './types';


// TODO: flip this around - create classes for each analysis type
// classes should provide:
//  + methods for default parameters
//  + creation of a job from current parameters
function getAnalysisDetails(analysisType: AnalysisTypes): AnalysisDetails {
    switch (analysisType) {
        case AnalysisTypes.APPLY_DISK_MASK: {
            return {
                type: analysisType,
                parameters: {
                    shape: "disk",
                    cx: 64,
                    cy: 64,
                    r: 15,
                }
            };
        }
        case AnalysisTypes.APPLY_RING_MASK: {
            return {
                type: analysisType,
                parameters: {
                    shape: "ring",
                    cx: 64,
                    cy: 64,
                    ri: 15,
                    ro: 50,
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
    // TODO: extract sensor information from dataset and use it for default parameter creation
    /*

    function selectDataset(state: RootReducer, dataset: string) {
        return state.dataset.byId[dataset];
    }

    const dataset = yield select(selectDataset, action.payload.dataset)
    const sensorWidth = 128;
    const sensorHeight = 128;
    */
    const analysis = {
        id: uuid(),
        dataset: action.payload.dataset,
        details: getAnalysisDetails(action.payload.analysisType),
        currentJob: "",
    }
    yield put(analysisActions.Actions.created(analysis))
}

function selectAnalysis(state: RootReducer, id: string) {
    return state.analyses.byId[id];
}

export function* runAnalysisSaga(action: ReturnType<typeof analysisActions.Actions.run>) {
    const analysis: Analysis = yield select(selectAnalysis, action.payload.id)
    const masks = [
        analysis.details.parameters
    ];
    const jobId = uuid();
    const job = yield call(startJob, jobId, analysis.dataset, masks);
    return yield put(analysisActions.Actions.running(
        action.payload.id,
        job.job,
    ))
}

export function* analysisRootSaga() {
    yield takeEvery(analysisActions.ActionTypes.CREATE, createAnalysisSaga);
    yield takeEvery(analysisActions.ActionTypes.RUN, runAnalysisSaga);
}