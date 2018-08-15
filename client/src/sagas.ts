import { all, call } from 'redux-saga/effects';
import { analysisRootSaga } from './analysis/sagas';
import { directoryListingSaga } from './browser/sagas';
import { webSocketSaga } from './channel/sagas';
import { clusterConnectionSaga } from './cluster/sagas';
import { getConfigOnReconnect, getConfigSaga, updateLocalConfigSaga } from './config/sagas';
import { datasetRootSaga } from './dataset/sagas';

export function* rootSaga() {
    yield call(getConfigSaga);
    yield all([
        getConfigOnReconnect(),
        webSocketSaga(),
        analysisRootSaga(),
        datasetRootSaga(),
        clusterConnectionSaga(),
        directoryListingSaga(),
        updateLocalConfigSaga(),
    ]);
}