import { all, call, put, take, takeEvery } from "redux-saga/effects";
import { v4 as uuid } from 'uuid';
import * as channelActions from '../channel/actions';
import { ConnectResponse } from "../messages";
import * as clusterActions from './actions';
import { checkClusterConnection, connectToCluster } from "./api";

function* connectSaga(action: ReturnType<typeof clusterActions.Actions.connect>) {
    yield put(clusterActions.Actions.connecting())
    const conn = (yield call(connectToCluster, action.payload.params)) as ConnectResponse;
    yield call(putClusterStatus, conn);
}

function* putClusterStatus(conn: ConnectResponse) {
    if (conn.status === "ok") {
        yield put(clusterActions.Actions.connected(conn.connection.connection));
    } else if (conn.status === "error") {
        yield put(clusterActions.Actions.notConnected());
        const timestamp = Date.now();
        const id = uuid();
        yield put(clusterActions.Actions.error(`error connecting to cluster: ${conn.msg}`,timestamp, id));
    } else {
        yield put(clusterActions.Actions.notConnected());
    }
}

/**
 * when the channel is connected, check if cluster is connected and update status accordingly
 */
function* trackClusterConnection() {
    while (true) {
        yield take(channelActions.ActionTypes.OPEN)
        const conn = (yield call(checkClusterConnection)) as ConnectResponse;
        yield call(putClusterStatus, conn);
    }
}

export function* clusterConnectionSaga() {
    yield takeEvery(clusterActions.ActionTypes.CONNECT, connectSaga);
    yield all([
        trackClusterConnection(),
    ])
}