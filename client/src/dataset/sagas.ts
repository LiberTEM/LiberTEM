import { call, put, takeEvery } from 'redux-saga/effects';
import { OpenDatasetResponse } from '../messages';
import * as datasetActions from "./actions";
import { openDataset } from './api';


export function* createDatasetSaga(action: ReturnType<typeof datasetActions.Actions.create>) {
    const resp: OpenDatasetResponse = yield call(openDataset, action.payload.dataset.id, { dataset: action.payload.dataset });
    yield put(datasetActions.Actions.created(resp.details));
}

export function* datasetRootSaga() {
    yield takeEvery(datasetActions.ActionTypes.CREATE, createDatasetSaga);
}