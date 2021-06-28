import { ActionsUnion, createAction } from "../helpers/actionHelpers";

export enum ActionTypes {
    GENERIC = "ERROR_GENERIC",
    DISMISS = "ERROR_DISMISS",
    DISMISS_ALL = "ERROR_DISMISS_ALL",
}

export const Actions = {
    dismiss: (id: string) => createAction(ActionTypes.DISMISS, { id }),
    dismissAll: () => createAction(ActionTypes.DISMISS_ALL),
    generic: (id: string, msg: string, timestamp: number) => createAction(ActionTypes.GENERIC, { id, msg, timestamp }),
}

export type Actions = ActionsUnion<typeof Actions>;