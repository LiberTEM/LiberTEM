import { call, put, select, takeEvery } from 'redux-saga/effects';
import { v4 as uuid } from 'uuid';
import * as browserActions from '../browser/actions';
import { joinPaths } from '../config/helpers';
import { ConfigState } from '../config/reducers';
import { DatasetFormInfo, DatasetFormParams, DetectDatasetResponse, OpenDatasetResponse } from '../messages';
import { RootReducer } from '../store';
import * as datasetActions from "./actions";
import { deleteDataset, detectDataset, openDataset } from './api';
import { isKnownDatasetType } from './helpers';


export function* createDatasetSaga(action: ReturnType<typeof datasetActions.Actions.create>) {
    try {
        const resp = (yield call(openDataset, action.payload.dataset.id, { dataset: action.payload.dataset })) as OpenDatasetResponse;
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
        yield put(datasetActions.Actions.error(action.payload.dataset.id, `Error loading dataset: ${(e as Error).toString()}`, timestamp, id));
    }
}

export function* deleteDatasetSaga(action: ReturnType<typeof datasetActions.Actions.delete>) {
    try {
        yield call(deleteDataset, action.payload.dataset);
    } catch (e) {
        const timestamp = Date.now();
        const id = uuid();
        yield put(datasetActions.Actions.error(action.payload.dataset, `Error closing dataset: ${(e as Error).toString()}`, timestamp, id));
    }
}

interface DetectResults {
    detectedParams?: DatasetFormParams,
    shouldOpen: boolean,
    detectedInfo?: DatasetFormInfo,
}

export function* doDetectDataset(fullPath: string): Generator<unknown, DetectResults, unknown> {
    yield put(datasetActions.Actions.detect(fullPath));
    const detectResult = (yield call(detectDataset, fullPath)) as DetectDatasetResponse;
    let detectedParams;
    let detectedInfo;
    let shouldOpen = true;
    if (detectResult.status === "ok") {
        if (isKnownDatasetType(detectResult.datasetParams.type)) {
          detectedParams = detectResult.datasetParams;
          detectedInfo = detectResult.datasetInfo;
          yield put(datasetActions.Actions.detected(fullPath, detectedParams, detectedInfo));
        }
        else {
          const timestamp = Date.now();
          const id = uuid();
          yield put(datasetActions.Actions.detectFailed(fullPath));
          shouldOpen = false;
          yield put(datasetActions.Actions.error(id, detectResult.datasetParams.type + ` dataset type is currently not supported in the GUI`, timestamp, id));
        }
    } else {
        yield put(datasetActions.Actions.detectFailed(fullPath));
    }
    return {detectedParams, shouldOpen, detectedInfo};
}

export function* doOpenDataset(fullPath: string) {
    const config = (yield select((state: RootReducer) => state.config)) as ConfigState;
    const cachedParams = config.lastOpened[fullPath];
    try {
        const { detectedParams, shouldOpen, detectedInfo } = (yield call(doDetectDataset, fullPath)) as DetectResults;
        if (shouldOpen) {
            yield put(datasetActions.Actions.open(fullPath, cachedParams, detectedParams, detectedInfo));
        }
    } catch (e) {
        yield put(datasetActions.Actions.detectFailed(fullPath));
    }
}

export function* openDatasetSagaFullPath(action: ReturnType<typeof browserActions.Actions.selectFullPath>) {
    const fullPath = action.payload.path;
    yield call(doOpenDataset, fullPath);
}

export function* openDatasetSaga(action: ReturnType<typeof browserActions.Actions.select>) {
    const config = (yield select((state: RootReducer) => state.config)) as ConfigState;
    const fullPath = joinPaths(config, action.payload.path, action.payload.name);
    yield call(doOpenDataset, fullPath);
}

export function* datasetRootSaga() {
    yield takeEvery(datasetActions.ActionTypes.CREATE, createDatasetSaga);
    yield takeEvery(datasetActions.ActionTypes.DELETE, deleteDatasetSaga);
    yield takeEvery(browserActions.ActionTypes.SELECT, openDatasetSaga);
    yield takeEvery(browserActions.ActionTypes.SELECT_FULL_PATH, openDatasetSagaFullPath);
}
