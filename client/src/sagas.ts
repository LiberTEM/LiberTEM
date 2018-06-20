import { all, call, takeEvery } from 'redux-saga/effects';
import * as channelActions from './channel/actions';
import { webSocketSaga } from './channel/sagas';
import { initialize } from './todo';


function* initSaga(action: ReturnType<typeof channelActions.Actions.initialState>) {
    yield call(initialize);
}


function* messageSaga() {
    yield takeEvery(channelActions.ActionTypes.INITIAL_STATE, initSaga);
}

export function* rootSaga() {
    yield all([
        messageSaga(),
        webSocketSaga(),
    ]);
}