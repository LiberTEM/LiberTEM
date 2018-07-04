import { all } from 'redux-saga/effects';
import { analysisRootSaga } from './analysis/sagas';
import { webSocketSaga } from './channel/sagas';
import { clusterConnectionSaga } from './cluster/sagas';
import { datasetRootSaga } from './dataset/sagas';

export function* rootSaga() {
    yield all([
        webSocketSaga(),
        analysisRootSaga(),
        datasetRootSaga(),
        clusterConnectionSaga(),
    ]);
}