import { createAction } from "../helpers/actionHelpers";

export enum ActionTypes {
    CREATE = 'JOB_CREATE',
}

export const Actions = {
    create: (id: string) => createAction(ActionTypes.CREATE, { id }),
}