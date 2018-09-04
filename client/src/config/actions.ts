import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { ConfigState } from "./reducers";

export enum ActionTypes {
    FETCHED = "CONFIG_FETCHED",
    FETCH = "CONFIG_FETCH",
}

export const Actions = {
    fetch: () => createAction(ActionTypes.FETCH),
    fetched: (config: ConfigState) => createAction(ActionTypes.FETCHED, { config }),
}

export type Actions = ActionsUnion<typeof Actions>;