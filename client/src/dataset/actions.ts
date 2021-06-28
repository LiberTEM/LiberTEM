import { ActionsUnion, createAction } from '../helpers/actionHelpers';
import { CreateDatasetMessage, DatasetCreateParams, DatasetFormInfo, DatasetFormParams } from '../messages';

export enum ActionTypes {
    OPEN = 'DATASET_OPEN',
    CANCEL_OPEN = 'DATASET_CANCEL_OPEN',
    CREATE = 'DATASET_CREATE',
    CREATED = 'DATASET_CREATED',
    ERROR = 'DATASET_ERROR',
    DELETE = 'DATASET_DELETE',
    DELETED = 'DATASET_DELETED',
    DETECT = 'DATASET_DETECT',
    DETECTED = 'DATASET_DETECTED',
    DETECT_FAILED = 'DATASET_DETECT_FAILED',
}

export const Actions = {
    open: (path: string, cachedParams?: DatasetFormParams, detectedParams?: DatasetFormParams, detectedInfo?: DatasetFormInfo) => createAction(ActionTypes.OPEN, { path, cachedParams, detectedParams, detectedInfo }),
    cancelOpen: () => createAction(ActionTypes.CANCEL_OPEN),
    create: (dataset: DatasetCreateParams) => createAction(ActionTypes.CREATE, { dataset }),
    created: (dataset: CreateDatasetMessage) => createAction(ActionTypes.CREATED, { dataset }),
    error: (dataset: string, msg: string, timestamp: number, id: string) => createAction(ActionTypes.ERROR, { dataset, msg, timestamp, id }),
    delete: (dataset: string) => createAction(ActionTypes.DELETE, { dataset }),
    deleted: (dataset: string) => createAction(ActionTypes.DELETED, { dataset }),
    detect: (path: string) => createAction(ActionTypes.DETECT, { path }),
    detected: (path: string, params: DatasetFormParams, info: DatasetFormInfo) => createAction(ActionTypes.DETECTED, { path, params, info }),
    detectFailed: (path: string) => createAction(ActionTypes.DETECT_FAILED, { path }),
}

export type Actions = ActionsUnion<typeof Actions>;
