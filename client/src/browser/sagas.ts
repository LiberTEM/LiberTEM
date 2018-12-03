import { call, fork, put, select, take } from "redux-saga/effects";
import * as uuid from 'uuid/v4';
import { joinPaths } from "../config/helpers";
import { ConfigState } from "../config/reducers";
import { DirectoryListingResponse } from "../messages";
import { RootReducer } from "../store";
import * as browserActions from './actions';
import { getDirectoryListing } from "./api";

export function* directoryListingSaga() {
    yield fork(fetchOnRequest);
    yield fork(fetchOnRequestFullPath);
    yield fork(fetchDirectoryListOnOpen);
}

function* fetchOnRequest() {
    while (true) {
        const action: ReturnType<typeof browserActions.Actions.list> = yield take(browserActions.ActionTypes.LIST_DIRECTORY);

        const { name, path } = action.payload;
        const config: ConfigState = yield select((state: RootReducer) => state.config)
        const newPath = name !== undefined ? joinPaths(config, path, name) : path;
        yield fork(fetchDirectoryListing, newPath);
    }
}

function* fetchOnRequestFullPath() {
    while (true) {
        const action: ReturnType<typeof browserActions.Actions.listFullPath> = yield take(browserActions.ActionTypes.LIST_FULL_PATH);
        yield fork(fetchDirectoryListing, action.payload.path);
    }
}

function* fetchDirectoryListing(path: string) {
    try {
        const result: DirectoryListingResponse = yield call(getDirectoryListing, path);
        if (result.status === "ok") {
            yield put(browserActions.Actions.dirListing(result.path, result.dirs, result.files, result.drives, result.places));
        } else if (result.status === "error") {
            const config: ConfigState = yield select((state: RootReducer) => state.config)
            const timestamp = Date.now();
            const id = uuid();
            yield put(browserActions.Actions.error(`Error browsing directory: ${result.msg}`, timestamp, id));
            yield put(browserActions.Actions.list(config.separator));
        }
    } catch (e) {
        const timestamp = Date.now();
        const id = uuid();
        yield put(browserActions.Actions.error(`Error browsing directory: ${e.toString()}`, timestamp, id));
    }
}

function* fetchDirectoryListOnOpen() {
    while (true) {
        yield take(browserActions.ActionTypes.OPEN);
        const config: ConfigState = yield select((state: RootReducer) => state.config)
        yield put(browserActions.Actions.list(config.cwd));
    }
}