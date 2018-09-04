import { ActionsUnion, createAction } from '../helpers/actionHelpers';
import { DatasetCreateParams, DatasetFormParams, DatasetState } from '../messages';

export enum ActionTypes {
    OPEN = 'DATASET_OPEN',
    CANCEL_OPEN = 'DATASET_CANCEL_OPEN',
    CREATE = 'DATASET_CREATE',
    CREATED = 'DATASET_CREATED',
    ERROR = 'DATASET_ERROR',
    DELETE = 'DATASET_DELETE',
    DELETED = 'DATASET_DELETED',
}

export const Actions = {
    open: (path: string, initialParams?: DatasetFormParams) => createAction(ActionTypes.OPEN, { path, initialParams }),
    cancelOpen: () => createAction(ActionTypes.CANCEL_OPEN),
    create: (dataset: DatasetCreateParams) => createAction(ActionTypes.CREATE, { dataset }),
    created: (dataset: DatasetState) => createAction(ActionTypes.CREATED, { dataset }),
    error: (dataset: string, msg: string, timestamp: number, id: string) => createAction(ActionTypes.ERROR, { dataset, msg, timestamp, id }),
    delete: (dataset: string) => createAction(ActionTypes.DELETE, { dataset }),
    deleted: (dataset: string) => createAction(ActionTypes.DELETED, { dataset }),
}

export type Actions = ActionsUnion<typeof Actions>;
