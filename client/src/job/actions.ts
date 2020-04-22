import { ActionsUnion, createAction } from "../helpers/actionHelpers";

export enum ActionTypes {
    CREATE = 'JOB_CREATE',
}

export const Actions = {
    create: (id: string, analysis: string, timestamp: number) => createAction(ActionTypes.CREATE, { id, analysis, timestamp }),
}

export type Actions = ActionsUnion<typeof Actions>;

export type ActionParts = {
    [K in keyof typeof Actions]: ReturnType<typeof Actions[K]>
}