import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { MsgPartConfig } from "../messages";

export enum ActionTypes {
    FETCHED = "CONFIG_FETCHED",
    FETCH = "CONFIG_FETCH",
}

export const Actions = {
    fetch: () => createAction(ActionTypes.FETCH),
    fetched: (config: MsgPartConfig) => createAction(ActionTypes.FETCHED, { config }),
}

export type Actions = ActionsUnion<typeof Actions>;