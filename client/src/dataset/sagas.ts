import { call, put, takeEvery } from 'redux-saga/effects';
import * as datasetActions from "./actions";
import { openDataset, OpenDatasetResponse } from './api';



export function* createDatasetSaga(action: ReturnType<typeof datasetActions.Actions.create>) {
    const resp: OpenDatasetResponse = yield call(openDataset, action.payload.dataset.id, action.payload.dataset);
    const dataset = {
        id: resp.dataset,
        ...action.payload.dataset,
    }
    yield put(datasetActions.Actions.created(dataset));
}

export function* datasetRootSaga() {
    yield takeEvery(datasetActions.Actions.create, createDatasetSaga);
}