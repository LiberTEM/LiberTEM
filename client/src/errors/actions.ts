import { ActionsUnion, createAction } from "../helpers/actionHelpers";

export enum ActionTypes {
    DISMISS = "ERROR_DISMISS",
    DISMISS_ALL = "ERROR_DISMISS_ALL",
}

export const Actions = {
    dismiss: (id: string) => createAction(ActionTypes.DISMISS, { id }),
    dismissAll: () => createAction(ActionTypes.DISMISS_ALL),
}

export type Actions = ActionsUnion<typeof Actions>;