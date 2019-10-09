import { call, put, select, takeEvery } from 'redux-saga/effects';
import uuid from 'uuid/v4';
import * as browserActions from '../browser/actions';
import { joinPaths } from '../config/helpers';
import { ConfigState } from '../config/reducers';
import { DetectDatasetResponse, OpenDatasetResponse } from '../messages';
import { RootReducer } from '../store';
import * as datasetActions from "./actions";
import { deleteDataset, detectDataset, openDataset } from './api';


export function* createDatasetSaga(action: ReturnType<typeof datasetActions.Actions.create>) {
    try {
        const resp: OpenDatasetResponse = yield call(openDataset, action.payload.dataset.id, { dataset: action.payload.dataset });
        if (resp.status === "ok") {
            yield put(datasetActions.Actions.created(resp.details));
        } else if (resp.status === "error") {
            const timestamp = Date.now();
            const id = uuid();
            yield put(datasetActions.Actions.error(resp.dataset, resp.msg, timestamp, id));
        }
    } catch (e) {
        const timestamp = Date.now();
        const id = uuid();
        yield put(datasetActions.Actions.error(action.payload.dataset.id, `Error loading dataset: ${e.toString()}`, timestamp, id));
    }
}

export function* deleteDatasetSaga(action: ReturnType<typeof datasetActions.Actions.delete>) {
    try {
        yield call(deleteDataset, action.payload.dataset);
    } catch (e) {
        const timestamp = Date.now();
        const id = uuid();
        yield put(datasetActions.Actions.error(action.payload.dataset, `Error closing dataset: ${e.toString()}`, timestamp, id));
    }
}

export function* doOpenDataset(fullPath: string) {
    const config: ConfigState = yield select((state: RootReducer) => state.config);
    const cachedParams = config.lastOpened[fullPath];
    let detectedParams;
    try {
        yield put(datasetActions.Actions.detect(fullPath));
        const detectResult: DetectDatasetResponse = yield call(detectDataset, fullPath);
        if (detectResult.status === "ok") {
            detectedParams = detectResult.datasetParams;
            yield put(datasetActions.Actions.detected(fullPath, detectResult.datasetParams));
        } else {
            yield put(datasetActions.Actions.detectFailed(fullPath));
        }
    } catch (e) {
        yield put(datasetActions.Actions.detectFailed(fullPath));
    }
    yield put(datasetActions.Actions.open(fullPath, cachedParams, detectedParams));
}

export function* openDatasetSagaFullPath(action: ReturnType<typeof browserActions.Actions.selectFullPath>) {
    const fullPath = action.payload.path;
    yield call(doOpenDataset, fullPath);
}

export function* openDatasetSaga(action: ReturnType<typeof browserActions.Actions.select>) {
    const config: ConfigState = yield select((state: RootReducer) => state.config);
    const fullPath = joinPaths(config, action.payload.path, action.payload.name);
    yield call(doOpenDataset, fullPath);
}

export function* datasetRootSaga() {
    yield takeEvery(datasetActions.ActionTypes.CREATE, createDatasetSaga);
    yield takeEvery(datasetActions.ActionTypes.DELETE, deleteDatasetSaga);
    yield takeEvery(browserActions.ActionTypes.SELECT, openDatasetSaga);
    yield takeEvery(browserActions.ActionTypes.SELECT_FULL_PATH, openDatasetSagaFullPath);
}