import { call, put, select, takeEvery } from 'redux-saga/effects';
import uuid from 'uuid/v4';
import * as browserActions from '../browser/actions';
import { ConfigState } from '../config/reducers';
import { DetectDatasetResponse, OpenDatasetResponse } from '../messages';
import { DirectoryListingDetails } from '../messages';
import { RootReducer } from '../store';
import * as datasetActions from "./actions";
import { deleteDataset, detectDataset, openDataset } from './api';
import { isKnownDatasetType } from './helpers';


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

export function* doDetectDataset(fullPath: string) {
    yield put(datasetActions.Actions.detect(fullPath.split(',')[0]));
    const detectResult: DetectDatasetResponse = yield call(detectDataset, fullPath);
    let detectedParams;
    let shouldOpen = true;
    if (detectResult.status === "ok") {
        if (isKnownDatasetType(detectResult.datasetParams.type)) {
          detectedParams = detectResult.datasetParams;
          yield put(datasetActions.Actions.detected(fullPath, detectResult.datasetParams));
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
    return [detectedParams, shouldOpen];
}

export function* doOpenDataset(fullPath: string) {
    const config: ConfigState = yield select((state: RootReducer) => state.config);
    const cachedParams = config.lastOpened[fullPath];
    let detectedParams;
    let shouldOpen = true;
    try {
      const doDetectDatasetRes = yield call(doDetectDataset, fullPath.split(',')[0]);
      detectedParams = doDetectDatasetRes[0];
      shouldOpen = doDetectDatasetRes[1];
    } catch (e) {
        yield put(datasetActions.Actions.detectFailed(fullPath));
    }
    if(shouldOpen) {
      yield put(datasetActions.Actions.open(fullPath, cachedParams, detectedParams));
    }
}

export function* openDatasetSagaFullPath(action: ReturnType<typeof browserActions.Actions.selectFullPath>) {
    const fullPath = action.payload.path;
    yield call(doOpenDataset, fullPath);
}

export function* openDatasetSaga(action: ReturnType<typeof browserActions.Actions.select>) {
    // const config: ConfigState = yield select((state: RootReducer) => state.config);
    // const fullPath = joinPaths(config, action.payload.path, action.payload.name);
    let files: DirectoryListingDetails[] = yield select((state:RootReducer)=>state.browser.files)
    const path: string = yield select((state:RootReducer)=>state.browser.path)
    files = files.filter(file => file.checked)
    let isValid = true;
    if(files.length > 1 ){
        const detectedFirstFileName: string = files[0].name.split('.').pop() as string;
        files.forEach(file => {
            const detectedFileName: string = file.name.split('.').pop() as string;
            if(detectedFileName !== detectedFirstFileName || 
                (detectedFileName !== 'dm3' && detectedFileName !== 'dm4')  ){
                    isValid = false;
            }
        })
    }
    if(!isValid){
        const timestamp = Date.now();
        const id = uuid();
        yield put(datasetActions.Actions.error(id,` dataset stack is not valid`, timestamp, id));
        return;
    }
    const fullPath = files.map(file => `${path}/${file.name}` ).toString()

    yield call(doOpenDataset, fullPath);
}

export function* datasetRootSaga() {
    yield takeEvery(datasetActions.ActionTypes.CREATE, createDatasetSaga);
    yield takeEvery(datasetActions.ActionTypes.DELETE, deleteDatasetSaga);
    yield takeEvery(browserActions.ActionTypes.SELECT_FILES, openDatasetSaga);
    yield takeEvery(browserActions.ActionTypes.SELECT_FULL_PATH, openDatasetSagaFullPath);
}
