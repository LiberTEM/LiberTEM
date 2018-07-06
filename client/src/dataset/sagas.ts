import { call, put, takeEvery } from 'redux-saga/effects';
import { OpenDatasetResponse } from '../messages';
import * as datasetActions from "./actions";
import { openDataset } from './api';


export function* createDatasetSaga(action: ReturnType<typeof datasetActions.Actions.create>) {
    try {
        const resp: OpenDatasetResponse = yield call(openDataset, action.payload.dataset.id, { dataset: action.payload.dataset });
        if (resp.status === "ok") {
            yield put(datasetActions.Actions.created(resp.details));
        } else if (resp.status === "error") {
            const timestamp = Date.now();
            yield put(datasetActions.Actions.error(resp.dataset, resp.msg, timestamp));
        }
    } catch (e) {
        const timestamp = Date.now();
        yield put(datasetActions.Actions.error(action.payload.dataset.id, `Error loading dataset: ${e.toString()}`, timestamp));
    }
}

export function* datasetRootSaga() {
    yield takeEvery(datasetActions.ActionTypes.CREATE, createDatasetSaga);
}