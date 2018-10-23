import { call, fork, put, select, take, takeEvery } from 'redux-saga/effects';
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
        yield put(configActions.Actions.fetched(getDefaultLocalConfig(configResponse.config)));
    }
}

/**
 * when browsing in the file browser, update localStorage cwd config entry
 */
function* updateConfigCWD() {
    while (true) {
        const action: ReturnType<typeof browserActions.Actions.dirListing> = yield take(browserActions.ActionTypes.DIRECTORY_LISTING)
        const config: ConfigState = yield select((state: RootReducer) => state.config);
        const newConfig = Object.assign({}, config, { cwd: action.payload.path })
        setLocalStorage(newConfig);
    }
}

/**
 * when opening a dataset, update lastOpened config value
 */
function* updateLastOpenedConfig() {
    while (true) {
        const action: ReturnType<typeof datasetActions.Actions.create> = yield take(datasetActions.ActionTypes.CREATE);
        const config: ConfigState = yield select((state: RootReducer) => state.config);
        const newLastOpened = Object.assign({}, config.lastOpened, { [action.payload.dataset.params.path]: action.payload.dataset.params });
        const newConfig = Object.assign({}, config, { lastOpened: newLastOpened });
        setLocalStorage(newConfig);
    }
}

export function* firstConfigFetch() {
    try {
        yield call(getConfigSaga);
    } catch (e) {
        // tslint:disable-next-line:no-console
        console.error("failed to fetch config");
    }
}

export function* configRootSaga() {
    yield fork(firstConfigFetch);
    yield fork(getConfigOnReconnect);
    yield fork(updateConfigCWD);
    yield fork(updateLastOpenedConfig);
}