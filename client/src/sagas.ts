import { all } from 'redux-saga/effects';
import { directoryListingSaga } from './browser/sagas';
import { webSocketSaga } from './channel/sagas';
import { clusterConnectionSaga } from './cluster/sagas';
import { analysisRootSaga } from './compoundAnalysis/sagas';
import { configRootSaga } from './config/sagas';
import { datasetRootSaga } from './dataset/sagas';

export function* rootSaga() {
    yield all([
        configRootSaga(),
        webSocketSaga(),
        analysisRootSaga(),
        datasetRootSaga(),
        clusterConnectionSaga(),
        directoryListingSaga(),
    ]);
}