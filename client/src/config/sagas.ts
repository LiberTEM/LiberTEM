import { call, fork, put, select, take, takeEvery } from 'redux-saga/effects';
import uuid from 'uuid/v4';
import * as browserActions from '../browser/actions';
import * as channelActions from '../channel/actions';
import * as datasetActions from '../dataset/actions';
import { GetConfigResponse } from '../messages';
import { RootReducer } from '../store';
import * as configActions from './actions';
import { getConfig } from './api';
import { clearLocalStorage, getDefaultLocalConfig, mergeLocalStorage, setLocalStorage } from './helpers';
import { ConfigState } from './reducers';

function* getConfigOnReconnect() {
    yield takeEvery(channelActions.ActionTypes.OPEN, getConfigSaga);
}

/**
 * get config from server and try to merge in the localStorage config
 */
function* getConfigSaga() {
    yield put(configActions.Actions.fetch());
    const configResponse: GetConfigResponse = yield call(getConfig);
    try {
        const mergedConfig = mergeLocalStorage(configResponse.config);
        yield put(configActions.Actions.fetched(mergedConfig));
    } catch (e) {
        try {
            clearLocalStorage();
            // tslint:disable-next-line:no-empty
        } catch (e) { }
        const defaultConfig = Object.assign({}, configResponse.config, getDefaultLocalConfig(configResponse.config));
        yield put(configActions.Actions.fetched(defaultConfig));
    }
}

/**
 * update localStorage config on opening files or using the file browser
 */
function* updateLocalStorageConfig() {
    while (true) {
        yield take([
            datasetActions.ActionTypes.CREATE,
            browserActions.ActionTypes.DIRECTORY_LISTING
        ]);
        const config: ConfigState = yield select((state: RootReducer) => state.config);
        setLocalStorage(config);
    }
}

export function* firstConfigFetch() {
    try {
        yield call(getConfigSaga);
    } catch (e) {
        const timestamp = Date.now();
        const id = uuid();
        yield put(configActions.Actions.fetchFailed(`failed to fetch config: ${e.toString()}`, timestamp, id));
    }
}

export function* configRootSaga() {
    yield fork(firstConfigFetch);
    yield fork(getConfigOnReconnect);
    yield fork(updateLocalStorageConfig);
}