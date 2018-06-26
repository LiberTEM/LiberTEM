import { all } from 'redux-saga/effects';
import { analysisRootSaga } from './analysis/sagas';
import { webSocketSaga } from './channel/sagas';

export function* rootSaga() {
    yield all([
        webSocketSaga(),
        analysisRootSaga(),
    ]);
}