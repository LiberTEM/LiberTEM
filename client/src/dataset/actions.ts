import { ActionsUnion, createAction } from '../helpers/actionHelpers';
import { Dataset, DatasetCreateParams } from '../messages';

export enum ActionTypes {
    CREATE = 'DATASET_CREATE',
    CREATED = 'DATASET_CREATED',
}

export const Actions = {
    create: (dataset: DatasetCreateParams) => createAction(ActionTypes.CREATE, { dataset }),
    created: (dataset: Dataset) => createAction(ActionTypes.CREATED, { dataset }),
}

export type Actions = ActionsUnion<typeof Actions>;
