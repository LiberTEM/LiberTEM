import { ActionsUnion, createAction } from '../helpers/actionHelpers';
import { DatasetCreateParams, DatasetState } from '../messages';

export enum ActionTypes {
    CREATE = 'DATASET_CREATE',
    CREATED = 'DATASET_CREATED',
    ERROR = 'DATASET_ERROR',
}

export const Actions = {
    create: (dataset: DatasetCreateParams) => createAction(ActionTypes.CREATE, { dataset }),
    created: (dataset: DatasetState) => createAction(ActionTypes.CREATED, { dataset }),
    error: (dataset: string, msg: string, timestamp: number) => createAction(ActionTypes.ERROR, { dataset, msg, timestamp })
}

export type Actions = ActionsUnion<typeof Actions>;
