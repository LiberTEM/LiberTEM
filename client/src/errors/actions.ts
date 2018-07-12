import { ActionsUnion, createAction } from "../helpers/actionHelpers";

export enum ActionTypes {
    DISMISS = "ERROR_DISMISS",
}

export const Actions = {
    dismiss: (id: string) => createAction(ActionTypes.DISMISS, { id }),
}

export type Actions = ActionsUnion<typeof Actions>;