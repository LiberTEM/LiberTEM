import { call, fork, put, select, take, takeEvery } from 'redux-saga/effects';
import { v4 as uuid } from 'uuid';
import * as browserActions from '../browser/actions';
import * as channelActions from '../channel/actions';
import * as clusterActions from '../cluster/actions';
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
    const configResponse = (yield call(getConfig)) as GetConfigResponse;
    try {
        const mergedConfig = mergeLocalStorage(configResponse.config);
        yield put(configActions.Actions.fetched(mergedConfig));
    } catch (e) {
        try {
            clearLocalStorage();
            // eslint-disable-next-line @typescript-eslint/no-shadow
        } catch (e) {
            // ignore any errors clearing local storage...
        }
        const defaultConfig = Object.assign({}, configResponse.config, getDefaultLocalConfig());
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
            browserActions.ActionTypes.DIRECTORY_LISTING,
            clusterActions.ActionTypes.CONNECTED,
            configActions.ActionTypes.TOGGLE_STAR,
        ]);
        const config = (yield select((state: RootReducer) => state.config)) as ConfigState;
        setLocalStorage(config);
    }
}

export function* firstConfigFetch() {
    try {
        yield call(getConfigSaga);
    } catch (e) {
        const timestamp = Date.now();
        const id = uuid();
        yield put(configActions.Actions.fetchFailed(`failed to fetch config: ${(e as Error).toString()}`, timestamp, id));
    }
}

export function* configRootSaga() {
    yield fork(firstConfigFetch);
    yield fork(getConfigOnReconnect);
    yield fork(updateLocalStorageConfig);
}