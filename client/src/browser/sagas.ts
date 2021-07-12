import { call, fork, put, select, take } from "redux-saga/effects";
import { v4 as uuid } from 'uuid';
import { joinPaths } from "../config/helpers";
import { ConfigState } from "../config/reducers";
import { DirectoryListingResponse } from "../messages";
import { RootReducer } from "../store";
import * as browserActions from './actions';
import { getDirectoryListing } from "./api";
import { DirectoryBrowserState } from "./types";

export function* directoryListingSaga() {
    yield fork(fetchOnRequest);
    yield fork(fetchDirectoryListOnOpen);
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
