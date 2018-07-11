import { delay } from 'redux-saga';
import { call, put, take } from 'redux-saga/effects';
import * as channelActions from '../channel/actions';
import { GetConfigResponse } from '../messages';
import * as configActions from './actions';
import { getConfig } from './api';

export function* getConfigOnReconnect() {
    // TODO: handle failure of getConfigSaga here
    while (true) {
        yield take(channelActions.ActionTypes.CLOSE);
        yield take(channelActions.ActionTypes.OPEN);
        yield delay(1000);
        yield call(getConfigSaga);
    }
}

export function* getConfigSaga() {
    yield put(configActions.Actions.fetch());
    const config: GetConfigResponse = yield call(getConfig);
    yield put(configActions.Actions.fetched(config.config));
}