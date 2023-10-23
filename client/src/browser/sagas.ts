import { call, fork, put, select, take } from "redux-saga/effects";
import { v4 as uuid } from 'uuid';
import * as clusterActions from '../cluster/actions';
import { joinPaths } from "../config/helpers";
import { ConfigState } from "../config/reducers";
import { DirectoryListingResponse, StatResponse } from "../messages";
import { RootReducer } from "../store";
import * as browserActions from './actions';
import { getDirectoryListing, getPathStat } from "./api";
import { getUrlAction } from "./helpers";
import { DirectoryBrowserState } from "./types";

// root saga: fork additional sagas here!
export function* directoryListingSaga() {
    yield fork(fetchOnRequest);
    yield fork(fetchDirectoryListOnOpen);
    yield fork(actionOnConnect);
}

function* fetchOnRequest() {
    while (true) {
        const action = (yield take(browserActions.ActionTypes.LIST_DIRECTORY)) as ReturnType<typeof browserActions.Actions.list>;

        const { name, path } = action.payload;
        const config = (yield select((state: RootReducer) => state.config)) as ConfigState;
        const newPath = name !== undefined ? joinPaths(config, path, name) : path;
        yield fork(fetchDirectoryListing, newPath);
    }
}

function* fetchDirectoryListing(path: string) {
    try {
        const result = (yield call(getDirectoryListing, path)) as DirectoryListingResponse;
        if (result.status === "ok") {
            yield put(browserActions.Actions.dirListing(result.path, result.dirs, result.files, result.drives, result.places));
        } else if (result.status === "error") {
            const browserState = (yield select((state: RootReducer) => state.browser)) as DirectoryBrowserState;
            const timestamp = Date.now();
            const id = uuid();
            const alternative = result.alternative ? result.alternative : browserState.places.home.path;
            // Don't show an error, if it's due to last recent directory not being available
            const config = (yield select((state: RootReducer) => state.config)) as ConfigState;
            if (config.cwd !== path) {
                yield put(browserActions.Actions.error(`Error browsing directory: ${result.msg}`, timestamp, id));
            }
            yield put(browserActions.Actions.list(alternative));
        }
    } catch (e) {
        const browserState = (yield select((state: RootReducer) => state.browser)) as DirectoryBrowserState;
        const timestamp = Date.now();
        const id = uuid();
        yield put(browserActions.Actions.error(`Error browsing directory: ${(e as Error).toString()}`, timestamp, id));
        yield put(browserActions.Actions.list(browserState.places.home.path));
    }
}

function* fetchDirectoryListOnOpen() {
    while (true) {
        yield take(browserActions.ActionTypes.OPEN);
        const config = (yield select((state: RootReducer) => state.config)) as ConfigState;
        yield put(browserActions.Actions.list(config.cwd));
    }
}

function* actionOpenOnConnect(path: string) {
    const result = (yield call(getPathStat, path)) as StatResponse;
    if (result.status === "ok") {
        if(result.stat.isdir) {
            yield put(browserActions.Actions.open());
            yield put(browserActions.Actions.list(path));
        } else {
            yield put(browserActions.Actions.select(result.dirname, result.basename));
        }
    } else if (result.status === "error") {
        const timestamp = Date.now();
        const id = uuid();
        yield put(browserActions.Actions.error(`Could not stat path ${path}: ${result.msg}`, timestamp, id));
    }
}

function* actionOnConnect() {
    while (true) {
        // when connecting to the cluster...
        yield take(clusterActions.ActionTypes.CONNECTED);

        // check for "#action=open&path=..." fragment in the URL:
        const action = getUrlAction();

        // act on the given action:
        switch (action.action) {
            case 'open':
                yield fork(actionOpenOnConnect, action.path);
                break;

            case 'error':
                const timestamp = Date.now();
                const id = uuid();
                yield put(browserActions.Actions.error(action.msg, timestamp, id));
                break;

            case 'none':
                // do nothing.
                break;
        }
    }
}