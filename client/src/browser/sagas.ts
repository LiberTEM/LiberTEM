import { call, fork, put, select, take } from "redux-saga/effects";
import * as uuid from 'uuid/v4';
import { ConfigState } from "../config/reducers";
import { DirectoryListingResponse } from "../messages";
import { RootReducer } from "../store";
import * as browserActions from './actions';
import { getDirectoryListing } from "./api";

export function* directoryListingSaga() {
    yield fork(fetchDirectoryListing);
    yield fork(fetchDirectoryListOnOpen);
}

function* fetchDirectoryListing() {
    while (true) {
        const action: ReturnType<typeof browserActions.Actions.list> = yield take(browserActions.ActionTypes.LIST_DIRECTORY);
        try {
            const { name, path } = action.payload;
            const config: ConfigState = yield select((state: RootReducer) => state.config)
            const newPath = name !== undefined ? `${path}${config.separator}${name}` : path;
            const result: DirectoryListingResponse = yield call(getDirectoryListing, newPath);
            yield put(browserActions.Actions.dirListing(result.path, result.dirs, result.files));
        } catch (e) {
            const timestamp = Date.now();
            const id = uuid();
            yield put(browserActions.Actions.error(`Error browsing directory: ${e.toString()}`, timestamp, id));
        }
    }
}

function* fetchDirectoryListOnOpen() {
    while (true) {
        yield take(browserActions.ActionTypes.OPEN);
        const config: ConfigState = yield select((state: RootReducer) => state.config)
        yield put(browserActions.Actions.list(config.cwd));
    }
}