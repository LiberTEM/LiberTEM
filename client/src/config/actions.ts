import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { ConfigState } from "./reducers";

export enum ActionTypes {
    FETCHED = "CONFIG_FETCHED",
    FETCH = "CONFIG_FETCH",
    FETCH_FAILED = "CONFIG_FETCH_FAILED",
}

export const Actions = {
    fetch: () => createAction(ActionTypes.FETCH),
    fetched: (config: ConfigState) => createAction(ActionTypes.FETCHED, { config }),
    fetchFailed: (msg: string, timestamp: number, id: string) => createAction(ActionTypes.FETCH_FAILED, { msg, timestamp, id }),
}

export type Actions = ActionsUnion<typeof Actions>;