import { delay } from 'redux-saga';
import { call, fork, put, select, take } from 'redux-saga/effects';
import * as browserActions from '../browser/actions';
import * as channelActions from '../channel/actions';
import * as datasetActions from '../dataset/actions';
import { GetConfigResponse } from '../messages';
import { RootReducer } from '../store';
import * as configActions from './actions';
import { getConfig } from './api';
import { clearLocalStorage, getDefaultLocalConfig, mergeLocalStorage, setLocalStorage } from './helpers';
import { ConfigState } from './reducers';

export function* getConfigOnReconnect() {
    // TODO: handle failure of getConfigSaga here
    while (true) {
        yield take(channelActions.ActionTypes.CLOSE);
        yield take(channelActions.ActionTypes.OPEN);
        yield delay(1000);
        yield call(getConfigSaga);
    }
}

/**
 * get config from server and try to merge in the localStorage config
 */
export function* getConfigSaga() {
    yield put(configActions.Actions.fetch());
    const config: GetConfigResponse = yield call(getConfig);
    try {
        const mergedConfig = mergeLocalStorage(config.config);
        yield put(configActions.Actions.fetched(mergedConfig));
    } catch (e) {
        try {
            clearLocalStorage();
            // tslint:disable-next-line:no-empty
        } catch (e) { }
        yield put(configActions.Actions.fetched(getDefaultLocalConfig(config.config)));
    }
}

export function* updateLocalConfigSaga() {
    yield fork(updateConfigCWD);
    yield fork(updateLastOpenedConfig);
}

/**
 * when browsing in the file browser, update localStorage cwd config entry
 */
export function* updateConfigCWD() {
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
export function* updateLastOpenedConfig() {
    while (true) {
        const action: ReturnType<typeof datasetActions.Actions.create> = yield take(datasetActions.ActionTypes.CREATE);
        const config: ConfigState = yield select((state: RootReducer) => state.config);
        const newLastOpened = Object.assign({}, config.lastOpened, { [action.payload.dataset.params.path]: action.payload.dataset.params });
        const newConfig = Object.assign({}, config, { lastOpened: newLastOpened });
        setLocalStorage(newConfig);
    }
}